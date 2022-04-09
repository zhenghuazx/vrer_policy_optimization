import scipy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def logprobabilities(action_probs, a, num_actions):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    action_mean, action_stdev = action_probs
    normal = tfp.distributions.Normal(action_mean, action_stdev, allow_nan_stats=False)
    logprobability = normal.log_prob(a)
    # logprobabilities_all = tf.math.log(action_probs)
    # logprobability = tf.reduce_sum(
    #     tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    # )
    return logprobability


def get_experience_replay_set(num_removed, buffer, action_probs_history_full, state_history_full, action_history_full, returns_history_full, advantage_buffer_full):
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, return_buffer = [], [], [], [], []
    for i, j in buffer:
        observation_buffer.append(state_history_full[i-num_removed][j])
        action_buffer.append(action_history_full[i-num_removed][j])
        logprobability_buffer.append(action_probs_history_full[i-num_removed][j])
        advantage_buffer.append(advantage_buffer_full[i-num_removed][j])
        return_buffer.append(returns_history_full[i-num_removed][j])
    advantage_mean, advantage_std = np.mean(advantage_buffer), np.std(advantage_buffer)
    advantage_buffer = (advantage_buffer - advantage_mean) / advantage_std
    return observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, return_buffer