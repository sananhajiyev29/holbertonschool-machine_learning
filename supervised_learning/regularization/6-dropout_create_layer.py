#!/usr/bin/env python3
"""Module that creates a neural network layer using dropout."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer of a neural network using dropout.

    Args:
        prev: tensor containing the output of the previous layer.
        n: the number of nodes the new layer should contain.
        activation: the activation function for the new layer.
        keep_prob: the probability that a node will be kept.
        training: boolean indicating whether the model is in training mode.

    Returns:
        The output of the new layer.
    """
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer='glorot_uniform'
    )(prev)
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    return dropout(layer, training=training)
