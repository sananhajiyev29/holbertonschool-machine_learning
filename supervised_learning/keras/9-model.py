#!/usr/bin/env python3
"""Module that saves and loads an entire Keras model."""

import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model.

    Args:
        network: the model to save.
        filename: the path of the file that the model should be saved to.

    Returns:
        None.
    """
    network.save(filename)


def load_model(filename):
    """Loads an entire model.

    Args:
        filename: the path of the file that the model should be loaded from.

    Returns:
        The loaded model.
    """
    return K.models.load_model(filename)
