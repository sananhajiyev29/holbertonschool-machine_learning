#!/usr/bin/env python3
"""Module that saves and loads a Keras model's configuration."""

import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format.

    Args:
        network: the model whose configuration should be saved.
        filename: the path of the file that the configuration should be saved.

    Returns:
        None.
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """Loads a model with a specific configuration.

    Args:
        filename: the path of the file containing the model's configuration.

    Returns:
        The loaded model.
    """
    with open(filename, 'r') as f:
        config = f.read()
    return K.models.model_from_json(config)
