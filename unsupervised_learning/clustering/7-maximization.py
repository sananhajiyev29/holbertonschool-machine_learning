#!/usr/bin/env python3
"""Module for calculating the maximization step in a GMM."""
import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        g: numpy.ndarray of shape (k, n) containing posterior probabilities.

    Returns:
        pi, m, S, or None, None, None on failure.
        pi is a numpy.ndarray of shape (k,) containing updated priors.
        m is a numpy.ndarray of shape (k, d) containing updated means.
        S is a numpy.ndarray of shape (k, d, d) containing updated covariances.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(g, np.ndarray) or len(g.shape) != 2):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    if n == 0 or d == 0 or k == 0 or g.shape[1] != n:
        return None, None, None

    n_k = np.sum(g, axis=1)
    if np.any(n_k == 0):
        return None, None, None

    pi = n_k / n
    m = np.matmul(g, X) / n_k[:, np.newaxis]
    diff = X[np.newaxis, :, :] - m[:, np.newaxis, :]
    S = np.matmul((g[:, :, np.newaxis] * diff).transpose(0, 2, 1), diff)
    S = S / n_k[:, np.newaxis, np.newaxis]

    return pi, m, S
