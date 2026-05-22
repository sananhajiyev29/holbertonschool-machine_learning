#!/usr/bin/env python3
"""Module for calculating the maximization step in a GMM."""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing the posterior
           probabilities for each data point in each cluster

    Returns:
        pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,) containing the updated priors
        m: numpy.ndarray of shape (k, d) containing the updated centroid means
        S: numpy.ndarray of shape (k, d, d) containing the updated
           covariance matrices
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2):
        return None, None, None

    if (not isinstance(g, np.ndarray) or len(g.shape) != 2):
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n == 0 or d == 0 or k == 0 or n_g != n:
        return None, None, None

    if np.any(g < 0):
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), np.ones(n)):
        return None, None, None

    n_k = np.sum(g, axis=1)

    if np.any(n_k == 0):
        return None, None, None

    pi = n_k / n

    m = np.matmul(g, X) / n_k[:, np.newaxis]

    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]
        weighted = g[i][:, np.newaxis] * diff
        S[i] = np.matmul(weighted.T, diff) / n_k[i]

    return pi, m, S
