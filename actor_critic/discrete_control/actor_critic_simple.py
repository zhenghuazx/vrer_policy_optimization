import os
import gym
from tensorflow.keras import layers
from tensorflow import keras
from actor_critic.discrete_control.selection_rule import *
huber_loss = keras.losses.Huber()


'''
Implement Actor Critic network
This network learns two functions:

Actor: This takes as input the state of our environment and returns a probability value for each action in its action space.
Critic: This takes as input the state of our environment and returns an estimate of total rewards in the future.
In our implementation, they share the initial layer.
'''
# problem
problem = 'Acrobot-v1'
# problem = 'MountainCar-v0'
# Configuration parameters for the whole setup
seed = 101
gamma = 0.99  # Discount factor for past rewards
# env = gym.make("CartPole-v0")  # Create the environment
env = gym.make(problem)
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
tf.experimental.numpy.random.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
lr = 0.001
if problem == 'CartPole-v0':
    env._max_episode_steps = 100
max_steps_per_episode = env._max_episode_steps

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])
_model_hist = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=lr)


num_episodes = 200

running_rewards_full = []
path = "./result/{}/actor_critic_simple/seed-{}-lr-{}".format(problem, seed, lr)

action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

score_history = []
pg_gradient_norm = []

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for _ in range(max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

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

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = tf.reduce_mean(actor_losses) + tf.reduce_mean(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        pg_normed_tf_grad = tf.sqrt(
            sum([tf.math.square(tf.norm(grads[k], ord='euclidean', axis=None, keepdims=None, name=None)) for k in
                 range(len(grads))]))
        pg_gradient_norm.append(pg_normed_tf_grad)
        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    avg_score = np.mean(score_history[-100:])
    if episode_count % 1 == 0:
        template = "current reward: {}, avg reward: {:.2f}, running reward: {:.2f} at episode {}; number of reuse: {}"
        print(template.format(episode_reward, avg_score, running_reward, episode_count, len([])))
    running_rewards_full.append(running_reward)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        # print("The new directory is created!")
    if avg_score > env.spec.reward_threshold - 5:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
    if episode_count >= num_episodes:  # Condition of termination
        break

np.save(path + "/score_history", score_history)
np.save(path + "/pg_gradient_norm", pg_gradient_norm)

