#!/usr/bin/env python3
"""Module for initializing K-means cluster centroids."""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer containing the number of clusters.

    Returns:
        numpy.ndarray of shape (k, d) containing initialized centroids,
        or None on failure.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0):
        return None

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    return np.random.uniform(minimum, maximum, size=(k, X.shape[1]))
