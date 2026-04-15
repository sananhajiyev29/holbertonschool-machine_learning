#!/usr/bin/env python3
"""Module that builds a transition layer."""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in Densely Connected
    Convolutional Networks.

    Args:
        X: output from the previous layer.
        nb_filters: integer representing the number of filters in X.
        compression: the compression factor for the transition layer.

    Returns:
        The output of the transition layer and the number of filters.
    """
    init = K.initializers.HeNormal(seed=0)
    nb_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        nb_filters, (1, 1), padding='same', kernel_initializer=init
    )(X)
    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(X)

    return X, nb_filters
