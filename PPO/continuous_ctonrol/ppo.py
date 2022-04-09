import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from PPO.continuous_ctonrol.optimizer import train_policy, train_value_function
from PPO.continuous_ctonrol.ppo_model import create_actor_model, create_critic_model
from PPO.continuous_ctonrol.selection_rule import compute_gradient, compute_likelihood
from PPO.continuous_ctonrol.util import mlp, get_experience_replay_set
from simulator.fermentation import mechanism, fermentation_env2

huber_loss = keras.losses.Huber()

# problem
problem = 'fermentation'
# Configuration parameters for the whole setup
seed = 47
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 300
process_model = mechanism()
env = fermentation_env2(process_model)
upper_bound = 0.02  # upper bounad of feeding
lower_bound = 0.0  # upper bounad of feeding

tf.random.set_seed(seed)
np.random.seed(seed)
tf.experimental.numpy.random.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

if problem == 'CartPole-v0':
    env._max_episode_steps = 100

num_inputs = env.num_observation
num_actions = env.num_action


# Hyperparameters of the PPO algorithm
epochs = 30
lr = 0.0001

clip_ratio = 0.2
policy_learning_rate = 1e-3
value_function_learning_rate = 5e-3
train_policy_iterations = 5
train_value_iterations = 5
lam = 0.98
target_kl = 0.01
hidden_sizes = 64

# Initialize the actor and the critic as keras models

actor, _actor = create_actor_model(num_inputs, num_actions, hidden_sizes, lower_bound, upper_bound,
                                   log_stdev_low=-10, log_stdev_high=5)

critic, _critic = create_critic_model(num_inputs, num_actions, hidden_sizes, lower_bound, upper_bound,
                                      log_stdev_low=-10, log_stdev_high=5)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Hyperparameters for variance reduced experience replay
num_episodes = 300
c = 1.5
batch_size = 512
running_rewards_full = []
window = 50
num_removed = 0

path = "./result/{}/ppo/c{}/seed-{}-plr-{}-vlr-{}-iter-{}-c-{}".format(problem, c, seed, policy_learning_rate,
                                                                            value_function_learning_rate,
                                                                            train_policy_iterations, c)

action_probs_history = []
critic_value_history = []
state_history = []
action_history = []
rewards_history = []
reuse_full = []
running_reward = 0
episode_count = 0

advantage_buffer_full = []
action_probs_history_full = []
critic_value_history_full = []
state_history_full = []
action_history_full = []
returns_history_full = []
model_history = []
score_history = []

pg_gradient_norm = []
mlr_gradient_norm = []
loglikelihoods = np.zeros((num_episodes, max_steps_per_episode, num_episodes))

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    reuses = []
    with tf.GradientTape() as tape:
        for _ in range(max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            state_history.append(state)
            # Predict action probabilities and estimated future rewards
            # from environment state
            action_mean, action_stdev = actor(state)
            critic_value = critic(state)
            critic_value_history.append(critic_value[0])

            # Sample action from action probability distribution
            normal = tfp.distributions.Normal(action_mean, action_stdev, allow_nan_stats=False)
            # normal = tfp.distributions.TruncatedNormal(
            #     action_mean, action_stdev, lower_bound, upper_bound, validate_args=False, allow_nan_stats=True,
            #     name='TruncatedNormal')
            action = normal.sample()
            action = tf.clip_by_value(action, lower_bound, upper_bound)[0]

            # gaussian likelihood
            log_pi = normal.log_prob(action)

            '''Equivalently
            pre_sum = -0.5 * (((action - action_mean) / (action_stdev + eps)) ** 2 + 2 * tf.math.log(action_stdev) + tf.math.log(2 * np.pi))
            log_pi = tf.reduce_sum(pre_sum, axis=1)
            '''

            action_probs_history.append(log_pi)
            action_history.append(action)

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward
            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.01 * episode_reward + (1 - 0.01) * running_reward
        score_history.append(episode_reward)
        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Advantage
        deltas = np.array(rewards_history + [0])[:-1] + gamma * np.array(critic_value_history + [0])[1:] - np.array(
            critic_value_history + [0])[:-1]
        advantage_buffer = []
        discounted_sum = 0
        for r in deltas[::-1]:
            discounted_sum = r + gamma * lam * discounted_sum
            advantage_buffer.insert(0, discounted_sum)
        advantage_buffer = np.array(advantage_buffer)
        advantage_buffer = advantage_buffer.tolist()
        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

        # Backpropagation
        loss_value = sum(actor_losses)
        grads = tape.gradient(loss_value, actor.trainable_variables)
        cur_normed_tf_grad = tf.sqrt(
            sum([tf.math.square(tf.norm(grads[i], ord='euclidean', axis=None, keepdims=None, name=None)) for i in
                 range(len(grads))]))
        pg_gradient_norm.append(cur_normed_tf_grad.numpy())

    action_probs_history_full.append(action_probs_history.copy())
    critic_value_history_full.append(critic_value_history.copy())
    state_history_full.append(state_history.copy())
    action_history_full.append(action_history.copy())
    returns_history_full.append(returns.copy())
    model_history.append(actor.get_weights())
    advantage_buffer_full.append(advantage_buffer.copy())
    while len(model_history) > window:
        action_probs_history_full.pop(0)
        critic_value_history_full.pop(0)
        state_history_full.pop(0)
        action_history_full.pop(0)
        returns_history_full.pop(0)
        model_history.pop(0)
        advantage_buffer_full.pop(0)
        num_removed += 1

    for i in range(max(episode_count - window + 1, 0), episode_count + 1):
        length_i_esisode = len(action_history_full[i - num_removed])
        for j in range(length_i_esisode):
            loglikelihoods[i, j, episode_count] = compute_likelihood(i - num_removed, j, episode_count - num_removed,
                                                                     action_history_full, state_history_full,
                                                                     model_history, _actor)

    for j in range(len(action_history_full[-1])):
        for p in range(max(episode_count - window + 1, 0), episode_count + 1):
            loglikelihoods[episode_count, j, p] = compute_likelihood(episode_count - num_removed, j, p - num_removed,
                                                                     action_history_full, state_history_full,
                                                                     model_history, _actor)

    # cur_normed_tf_grad = compute_current_gradient(model, action_probs_history, critic_value_history, returns)
    num_steps = len(action_probs_history)
    reuses = list(zip([episode_count] * num_steps, list(range(num_steps))))
    for i in range(max(episode_count - window + 1, 0), episode_count):
        length_i_esisode = action_history_full[i - num_removed]
        for j in range(len(length_i_esisode)):
            numerator = np.exp(loglikelihoods[i, j, episode_count])
            denominator = np.exp(loglikelihoods[i, j, i])
            tf_grad = compute_gradient(actor, i - num_removed, j, returns_history_full, action_history_full,
                                       state_history_full, critic_value_history_full)
            normed_grad = sum(
                [tf.math.square(tf.norm(tf_grad[k], ord='euclidean', axis=None, keepdims=None, name=None)) for k in
                 range(len(tf_grad))])
            expected_normed_ilr_grad = numerator / denominator * tf.sqrt(normed_grad)
            if expected_normed_ilr_grad < c * cur_normed_tf_grad:
                reuses.append([i, j])

    # retrieve the samples from the buffer
    observation_buffer, action_buffer, logprobability_buffer, advtg_buffer, return_buffer = \
        get_experience_replay_set(num_removed, reuses, action_probs_history_full, state_history_full,
                                  action_history_full, returns_history_full, advantage_buffer_full)
    mlr_normed_tf_grad = 0
    for _ in range(train_policy_iterations):
        kl, _grad_norm = train_policy(actor, clip_ratio, policy_optimizer, observation_buffer, action_buffer,
                                      logprobability_buffer, advtg_buffer, num_actions)
        mlr_normed_tf_grad += _grad_norm
        if kl > 1.5 * target_kl:
            # Early Stopping
            break
    mlr_gradient_norm.append(mlr_normed_tf_grad.numpy() / train_policy_iterations)  # averaged normed gradient
    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(critic, observation_buffer, return_buffer, value_optimizer)

    # Clear the loss and reward history
    action_probs_history.clear()
    critic_value_history.clear()
    rewards_history.clear()
    state_history.clear()
    action_history.clear()
    reuse_full.append(reuses)

    # Log details
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    episode_count += 1
    avg_score = np.mean(score_history[-100:])
    if episode_count % 1 == 0:
        template = "current reward: {}, avg reward: {:.2f}, running reward: {:.2f} at episode {}; number of reuse: {}"
        print(template.format(episode_reward, avg_score, running_reward, episode_count, len(reuses)))
    if avg_score >= 95:
        print("Solved at episode {}!".format(episode_count))
    if episode_count >= num_episodes:  # Condition to consider the task solved
        break
    running_rewards_full.append(running_reward)


np.save(path + '/mlr_gradient_norm', mlr_gradient_norm)
np.save(path + '/reuse_full', reuse_full)
np.save(path + "/score_history", score_history)
np.save(path + "/pg_gradient_norm", pg_gradient_norm)
