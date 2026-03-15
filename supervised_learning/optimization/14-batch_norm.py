#!/usr/bin/env python3
"""Module that creates a batch normalization layer in TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev: the activated output of the previous layer.
        n: the number of nodes in the layer to be created.
        activation: the activation function for the output of the layer.

    Returns:
        A tensor of the activated output for the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer
    )(prev)

    gamma = tf.Variable(
        tf.ones([n]),
        trainable=True,
        name='gamma'
    )
    beta = tf.Variable(
        tf.zeros([n]),
        trainable=True,
        name='beta'
    )

    mean, variance = tf.nn.moments(dense, axes=[0])

    Z_norm = tf.nn.batch_normalization(
        dense,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-7
    )

    return activation(Z_norm)
