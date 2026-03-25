#!/usr/bin/env python3
"""Module that creates a neural network layer with L2 regularization."""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a neural network layer with L2 regularization.

    Args:
        prev: tensor containing the output of the previous layer.
        n: the number of nodes the new layer should contain.
        activation: the activation function for the layer.
        lambtha: the L2 regularization parameter.

    Returns:
        The output of the new layer.
    """
    regularizer = tf.keras.regularizers.L2(lambtha * 2)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_initializer='glorot_uniform'
    )
    return layer(prev)
