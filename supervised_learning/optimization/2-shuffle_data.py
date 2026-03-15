#!/usr/bin/env python3
"""Module that shuffles data points in two matrices the same way."""

import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.

    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle.
        Y: numpy.ndarray of shape (m, ny) to shuffle.

    Returns:
        The shuffled X and Y matrices.
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
