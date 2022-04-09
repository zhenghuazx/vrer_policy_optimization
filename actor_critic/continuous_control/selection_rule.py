import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


# compute likelihood for [episode i, transition j, model p]
def compute_likelihood(i, j, p, action_history_full, state_history_full, model_history, _model_hist, lower_bound=0,
                       upper_bound=0.02):
    state = state_history_full[i][j]
    # state = tf.convert_to_tensor([state], dtype=tf.float32)
    _model_hist.set_weights(model_history[p])
    action_mean, action_stdev, _ = _model_hist(state)
    normal = tfp.distributions.Normal(action_mean, action_stdev, allow_nan_stats=False)
    # normal = tfp.distributions.TruncatedNormal(
    #     action_mean, action_stdev, lower_bound, upper_bound, validate_args=False, allow_nan_stats=True,
    #     name='TruncatedNormal')
    log_prob = normal.log_prob(action_history_full[i][j])
    cur_likelihood = tf.squeeze(log_prob)
    return cur_likelihood


def compute_loss(model, i, j, returns_history_full, action_history_full, state_history_full, model_history, huber_loss,
                 lower_bound=0, upper_bound=0.02):
    ret = returns_history_full[i][j]
    state = state_history_full[i][j]
    # state = tf.convert_to_tensor([state], dtype=tf.float32)
    action_mean, action_stdev, critic_value = model(state)
    diff = ret - critic_value  # advantage
    normal = tfp.distributions.Normal(action_mean, action_stdev, allow_nan_stats=False)
    # normal = tfp.distributions.TruncatedNormal(
    #     action_mean, action_stdev, lower_bound, upper_bound, validate_args=False, allow_nan_stats=True,
    #     name='TruncatedNormal')
    log_prob = normal.log_prob(action_history_full[i][j])
    actor_loss = -diff * tf.squeeze(log_prob)
    critic_loss = huber_loss(tf.expand_dims(critic_value, 0), tf.expand_dims(ret, 0))
    return actor_loss, critic_loss


def compute_gradient(model, i, j, returns_history_full, action_history_full, state_history_full, model_history,
                     huber_loss):
    with tf.GradientTape(persistent=True) as tape:
        actor_loss, critic_loss = compute_loss(model, i, j, returns_history_full, action_history_full,
                                               state_history_full, model_history, huber_loss)
        loss_value = tf.math.reduce_mean([actor_loss]) + tf.math.reduce_mean([critic_loss])
    return tape.gradient(loss_value, model.trainable_variables)


def compute_current_gradient(model, action_probs_history, critic_value_history, returns, huber_loss):
    with tf.GradientTape() as tape:
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
        loss_value = tf.math.reduce_mean(actor_losses) + tf.math.reduce_mean(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
    return tf.norm(grads, ord='euclidean', axis=None, keepdims=None, name=None)


def compute_mlr_gradient(model, buffer, returns_history_full, action_history_full, state_history_full, model_history,
                         loglikelihoods, huber_loss):
    with tf.GradientTape(persistent=True) as tape:
        actor_losses = []
        critic_losses = []
        reused_model = set([i for i, j in buffer])
        for i, j in buffer:
            numerator = np.exp(loglikelihoods[i, j, len(model_history) - 1])
            denominator = np.mean(
                np.exp(loglikelihoods[i, j, [k for k in list(reused_model) if i - 10 <= k <= i + 10]]))
            actor_loss, critic_loss = compute_loss(model, i, j, returns_history_full, action_history_full,
                                                   state_history_full, model_history, huber_loss)
            actor_losses.append(numerator / denominator * actor_loss)
            critic_losses.append(critic_loss)

        loss_value = tf.math.reduce_mean(actor_losses) + tf.math.reduce_mean(critic_losses)
    return tape.gradient(loss_value, model.trainable_variables)
