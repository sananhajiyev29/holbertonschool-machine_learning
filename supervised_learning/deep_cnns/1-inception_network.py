#!/usr/bin/env python3
"""Module that builds the inception network."""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network as described in Going Deeper with
    Convolutions (2014).

    Returns:
        The keras model.
    """
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', activation='relu'
    )(X)
    pool1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(conv1)

    conv2 = K.layers.Conv2D(
        64, (1, 1), padding='same', activation='relu'
    )(pool1)
    conv3 = K.layers.Conv2D(
        192, (3, 3), padding='same', activation='relu'
    )(conv2)
    pool2 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(conv3)

    inc3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inc3b = inception_block(inc3a, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(inc3b)

    inc4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inc4b = inception_block(inc4a, [160, 112, 224, 24, 64, 64])
    inc4c = inception_block(inc4b, [128, 128, 256, 24, 64, 64])
    inc4d = inception_block(inc4c, [112, 144, 288, 32, 64, 64])
    inc4e = inception_block(inc4d, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(inc4e)

    inc5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inc5b = inception_block(inc5a, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D(
        (7, 7), strides=(1, 1), padding='valid'
    )(inc5b)
    dropout = K.layers.Dropout(0.4)(avg_pool)
    output = K.layers.Dense(1000, activation='softmax')(dropout)

    return K.models.Model(inputs=X, outputs=output)
