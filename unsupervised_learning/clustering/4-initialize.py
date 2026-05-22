#!/usr/bin/env python3
"""Module for initializing Gaussian Mixture Model variables."""
import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: positive integer containing the number of clusters.

    Returns:
        pi, m, S, or None, None, None on failure.
        pi is a numpy.ndarray of shape (k,) containing cluster priors.
        m is a numpy.ndarray of shape (k, d) containing centroid means.
        S is a numpy.ndarray of shape (k, d, d) containing covariances.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            X.shape[0] == 0 or not isinstance(k, int) or k <= 0):
        return None, None, None

    pi = np.ones(k) / k
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    S = np.tile(np.eye(X.shape[1]), (k, 1, 1))

    return pi, m, S
