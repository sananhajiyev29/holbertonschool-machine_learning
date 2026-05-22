#!/usr/bin/env python3
"""Module that calculates gradients for t-SNE."""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculates the gradients of Y.

    Args:
        Y: numpy.ndarray of shape (n, ndim) containing the low
            dimensional transformation of X.
        P: numpy.ndarray of shape (n, n) containing the P affinities.

    Returns:
        Tuple of (dY, Q) where dY is the gradients of Y and Q is the
        Q affinities of Y.
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    PQ = P - Q
    dY = np.zeros((n, ndim))

    for i in range(n):
        dY[i] = np.sum(
            np.tile(PQ[:, i] * num[:, i], (ndim, 1)).T * (Y[i] - Y),
            axis=0
        )

    return dY, Q
