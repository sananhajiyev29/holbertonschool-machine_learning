#!/usr/bin/env python3
"""Module for finding the optimum number of K-means clusters."""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: positive integer containing the minimum number of clusters.
        kmax: positive integer containing the maximum number of clusters.
        iterations: positive integer containing the maximum number of
            iterations for K-means.

    Returns:
        results, d_vars, or None, None on failure.
        results is a list containing the K-means output for each cluster size.
        d_vars is a list containing the variance difference from kmin.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or X.shape[0] == 0:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or kmax <= kmin:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    base_var = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        var = variance(X, C)
        if base_var is None:
            base_var = var
        d_vars.append(base_var - var)

    return results, d_vars
