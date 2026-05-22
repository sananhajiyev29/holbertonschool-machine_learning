#!/usr/bin/env python3
"""Module with a K-means variance helper."""
import numpy as np


def variance(X, C):
    """Calculate the total intra-cluster variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        C: numpy.ndarray of shape (k, d) containing centroid means.

    Returns:
        The total variance, or None on failure.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(C, np.ndarray) or len(C.shape) != 2 or
            X.shape[0] == 0 or C.shape[0] == 0 or
            X.shape[1] != C.shape[1]):
        return None

    distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)

    return np.sum(np.min(distances, axis=1))
