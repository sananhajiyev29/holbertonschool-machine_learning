#!/usr/bin/env python3
"""Module that builds a dense block."""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely Connected
    Convolutional Networks.

    Args:
        X: output from the previous layer.
        nb_filters: integer representing the number of filters in X.
        growth_rate: the growth rate for the dense block.
        layers: the number of layers in the dense block.

    Returns:
        The concatenated output and the number of filters respectively.
    """
    init = K.initializers.HeNormal(seed=0)

    for _ in range(layers):
        X_bn = K.layers.BatchNormalization(axis=3)(X)
        X_bn = K.layers.Activation('relu')(X_bn)
        X_bn = K.layers.Conv2D(
            4 * growth_rate, (1, 1), padding='same',
            kernel_initializer=init
        )(X_bn)

        X_bn = K.layers.BatchNormalization(axis=3)(X_bn)
        X_bn = K.layers.Activation('relu')(X_bn)
        X_bn = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same',
            kernel_initializer=init
        )(X_bn)

        X = K.layers.concatenate([X, X_bn])
        nb_filters += growth_rate

    return X, nb_filters
