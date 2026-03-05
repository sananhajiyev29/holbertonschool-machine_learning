#!/usr/bin/env python3
"""Module that defines a one-hot encoding function."""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix.

    Args:
        Y: numpy.ndarray with shape (m,) containing numeric class labels.
        classes: maximum number of classes found in Y.

    Returns:
        One-hot encoding of Y with shape (classes, m), or None on failure.
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < 2:
        return None
    if classes < np.max(Y) + 1:
        return None
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
