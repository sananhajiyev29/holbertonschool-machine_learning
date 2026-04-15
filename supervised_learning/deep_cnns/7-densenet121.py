#!/usr/bin/env python3
"""Module that builds the DenseNet-121 architecture."""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture as described in Densely
    Connected Convolutional Networks.

    Args:
        growth_rate: the growth rate.
        compression: the compression factor.

    Returns:
        The keras model.
    """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    nb_filters = 2 * growth_rate

    Y = K.layers.BatchNormalization(axis=3)(X)
    Y = K.layers.Activation('relu')(Y)
    Y = K.layers.Conv2D(
        nb_filters, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=init
    )(Y)
    Y = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(Y)

    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 6)
    Y, nb_filters = transition_layer(Y, nb_filters, compression)

    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 12)
    Y, nb_filters = transition_layer(Y, nb_filters, compression)

    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 24)
    Y, nb_filters = transition_layer(Y, nb_filters, compression)

    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 16)

    Y = K.layers.AveragePooling2D(
        (7, 7), strides=(1, 1), padding='valid'
    )(Y)
    output = K.layers.Dense(
        1000, activation='softmax', kernel_initializer=init
    )(Y)

    return K.models.Model(inputs=X, outputs=output)
