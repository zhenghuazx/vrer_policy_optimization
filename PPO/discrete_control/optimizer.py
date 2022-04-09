import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import keras
from PPO.discrete_control.util import logprobabilities

huber_loss = keras.losses.Huber()



# @tf.function
def train_policy(actor,
                 clip_ratio,
                 policy_optimizer,
                 observation_buffer,
                 action_buffer,
                 logprobability_buffer,
                 advantage_buffer,
                 num_actions):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(tf.squeeze(tf.convert_to_tensor(observation_buffer))), action_buffer, num_actions)
            - logprobability_buffer
        )
        advantage_buffer = tf.cast(advantage_buffer, tf.float32)
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )
        ratio = tf.cast(ratio, tf.float32)
        min_advantage = tf.cast(min_advantage, tf.float32)
        # advantage_buffer = tf.convert_to_tensor(advantage_buffer)
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(tf.squeeze(tf.convert_to_tensor(observation_buffer))), action_buffer, num_actions)
    )
    kl = tf.reduce_sum(kl)

    # compute the policy gradient variance for analysis
    mlr_normed_tf_grad = tf.sqrt(
        sum([tf.math.square(tf.norm(policy_grads[k], ord='euclidean', axis=None, keepdims=None, name=None)) for k in
             range(len(policy_grads))]))
    return kl, mlr_normed_tf_grad


# Train the value function by regression on mean-squared error
# @tf.function
def train_value_function(critic, observation_buffer, return_buffer, value_optimizer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(tf.squeeze(tf.convert_to_tensor(observation_buffer)))) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))
