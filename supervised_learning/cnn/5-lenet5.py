#!/usr/bin/env python3
"""Module that builds a modified LeNet-5 architecture using Keras."""
from tensorflow import keras as K


def lenet5(X):
    """Builds a modified LeNet-5 architecture using Keras.

    Args:
        X: K.Input of shape (m, 28, 28, 1) containing input images.

    Returns:
        K.Model compiled with Adam optimizer and accuracy metrics.
    """
    init = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(
        6, (5, 5), padding='same', activation='relu',
        kernel_initializer=init
    )(X)
    pool1 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(
        16, (5, 5), padding='valid', activation='relu',
        kernel_initializer=init
    )(pool1)
    pool2 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    flat = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(
        120, activation='relu', kernel_initializer=init
    )(flat)
    fc2 = K.layers.Dense(
        84, activation='relu', kernel_initializer=init
    )(fc1)
    output = K.layers.Dense(
        10, activation='softmax', kernel_initializer=init
    )(fc2)

    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
