#!/usr/bin/env python3
"""Module for performing K-means clustering."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer containing the number of clusters.
        iterations: positive integer containing the maximum number of
            iterations to perform.

    Returns:
        C, clss, or None, None on failure.
        C is a numpy.ndarray of shape (k, d) containing the centroid means.
        clss is a numpy.ndarray of shape (n,) containing cluster indices.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)
    C = np.random.uniform(minimum, maximum, size=(k, X.shape[1]))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        new_C = C.copy()

        for cluster in range(k):
            points = X[clss == cluster]
            if points.shape[0] == 0:
                new_C[cluster] = np.random.uniform(
                    minimum, maximum, size=(X.shape[1],)
                )
            else:
                new_C[cluster] = np.mean(points, axis=0)

        if np.array_equal(new_C, C):
            return C, clss

        C = new_C

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
