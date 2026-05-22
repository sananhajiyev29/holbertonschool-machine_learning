#!/usr/bin/env python3
"""Module that performs PCA on a dataset to a given dimensionality."""
import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) where n is the number of data
            points and d is the number of dimensions in each point.
        ndim: the new dimensionality of the transformed X.

    Returns:
        T: numpy.ndarray of shape (n, ndim) containing the transformed
            version of X.
    """
    X_centered = X - np.mean(X, axis=0)

    u, s, vh = np.linalg.svd(X_centered)

    W = vh[:ndim].T

    T = np.matmul(X_centered, W)

    return T
