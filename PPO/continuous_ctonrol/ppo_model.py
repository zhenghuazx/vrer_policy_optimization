import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_actor_model(num_inputs, num_actions, num_hidden, lower_bound, upper_bound, log_stdev_low=-10,
                       log_stdev_high=5):
    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    mean = layers.Dense(num_actions, activation='sigmoid')(common)
    mean = mean * upper_bound
    log_stdev = layers.Dense(num_actions, activation='linear')(common)

    # constrain the standard deviation
    log_stdev = tf.clip_by_value(log_stdev, log_stdev_low, log_stdev_high)
    stdev = tf.exp(log_stdev)

    model = keras.Model(inputs=inputs, outputs=[mean, stdev])
    _model_hist = keras.Model(inputs=inputs, outputs=[mean, stdev])
    return model, _model_hist


def create_critic_model(num_inputs, num_actions, num_hidden, lower_bound, upper_bound, log_stdev_low=-20,
                        log_stdev_high=2):
    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    critic = layers.Dense(1)(common)

    model = keras.Model(inputs=inputs, outputs=critic)
    _model_hist = keras.Model(inputs=inputs, outputs=critic)
    return model, _model_hist
