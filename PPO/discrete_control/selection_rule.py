import tensorflow_probability as tfp
import tensorflow as tf


# compute likelihood for [episode i, transition j, model p]
def compute_likelihood(i, j, p, action_history_full, state_history_full, model_history, _actor):
    state = state_history_full[i][j]
    # state = tf.convert_to_tensor([state], dtype=tf.float32)
    _actor.set_weights(model_history[p])
    probs = _actor(state)
    action_probs = tfp.distributions.Categorical(probs=probs)
    log_prob = action_probs.log_prob(action_history_full[i][j])
    cur_likelihood = tf.squeeze(log_prob)

    return cur_likelihood


def compute_gradient(actor, i, j, returns_history_full, action_history_full, state_history_full,
                     critic_value_history_full):
    with tf.GradientTape(persistent=True) as tape:
        # actor_loss, critic_loss = compute_loss(model, i, j, returns_history_full, action_history_full, state_history_full, model_history)
        ret = returns_history_full[i][j]
        state = state_history_full[i][j]
        critic_value = critic_value_history_full[i][j]
        # state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = actor(state)
        diff = ret - critic_value  # advantage
        action_probs = tfp.distributions.Categorical(probs=probs)
        log_prob = action_probs.log_prob(action_history_full[i][j])
        actor_loss = -diff * tf.squeeze(log_prob)
        # critic_loss = huber_loss(tf.expand_dims(critic_value, 0), tf.expand_dims(ret, 0))
        loss_value = tf.reduce_mean([actor_loss])
    return tape.gradient(loss_value, actor.trainable_variables)
