#!/usr/bin/env python3
"""Module that builds the ResNet-50 architecture."""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015).

    Returns:
        The keras model.
    """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    C = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=init
    )(X)
    C = K.layers.BatchNormalization(axis=3)(C)
    C = K.layers.Activation('relu')(C)
    C = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(C)

    C = projection_block(C, [64, 64, 256], s=1)
    C = identity_block(C, [64, 64, 256])
    C = identity_block(C, [64, 64, 256])

    C = projection_block(C, [128, 128, 512], s=2)
    C = identity_block(C, [128, 128, 512])
    C = identity_block(C, [128, 128, 512])
    C = identity_block(C, [128, 128, 512])

    C = projection_block(C, [256, 256, 1024], s=2)
    C = identity_block(C, [256, 256, 1024])
    C = identity_block(C, [256, 256, 1024])
    C = identity_block(C, [256, 256, 1024])
    C = identity_block(C, [256, 256, 1024])
    C = identity_block(C, [256, 256, 1024])

    C = projection_block(C, [512, 512, 2048], s=2)
    C = identity_block(C, [512, 512, 2048])
    C = identity_block(C, [512, 512, 2048])

    C = K.layers.AveragePooling2D(
        (7, 7), strides=(1, 1), padding='valid'
    )(C)
    output = K.layers.Dense(
        1000, activation='softmax', kernel_initializer=init
    )(C)

    return K.models.Model(inputs=X, outputs=output)
