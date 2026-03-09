#!/usr/bin/env python3
"""Module that builds a neural network using the Keras Input class."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library.

    Args:
        nx: number of input features to the network.
        layers: list containing the number of nodes in each layer.
        activations: list containing the activation functions for each layer.
        lambtha: L2 regularization parameter.
        keep_prob: probability that a node will be kept for dropout.

    Returns:
        The keras model.
    """
    regularizer = K.regularizers.L2(lambtha)
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer
        )(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    return K.Model(inputs=inputs, outputs=x)
