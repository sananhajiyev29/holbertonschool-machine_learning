#!/usr/bin/env python3
"""Module that defines a one-hot decoding function."""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m).

    Returns:
        numpy.ndarray with shape (m,) containing numeric labels, or None.
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
