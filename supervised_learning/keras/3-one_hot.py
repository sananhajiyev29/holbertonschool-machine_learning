#!/usr/bin/env python3
"""Module that converts a label vector into a one-hot matrix."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix.

    Args:
        labels: numpy array of labels.
        classes: number of classes.

    Returns:
        The one-hot matrix.
    """
    if classes is None:
        classes = labels.max() + 1
    return K.utils.to_categorical(labels, num_classes=classes)
