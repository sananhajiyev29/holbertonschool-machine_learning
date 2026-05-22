#!/usr/bin/env python3
"""Module that calculates Q affinities for t-SNE."""
import numpy as np


def Q_affinities(Y):
    """Calculates the Q affinities.

    Args:
        Y: numpy.ndarray of shape (n, ndim) containing the low
            dimensional transformation of X.

    Returns:
        Q: numpy.ndarray of shape (n, n) containing the Q affinities.
        num: numpy.ndarray of shape (n, n) containing the numerator of
            the Q affinities.
    """
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)

    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)

    Q = num / np.sum(num)

    return Q, num
