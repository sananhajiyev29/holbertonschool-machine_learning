#!/usr/bin/env python3
"""Module that performs PCA on a dataset."""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) where n is the number of data
            points and d is the number of dimensions in each point.
            All dimensions have a mean of 0 across all data points.
        var: fraction of the variance that the PCA transformation
            should maintain.

    Returns:
        The weights matrix W of shape (d, nd) that maintains var
        fraction of X's original variance.
    """
    u, s, vh = np.linalg.svd(X)

    cumulative = np.cumsum(s) / np.sum(s)
    r = np.argwhere(cumulative >= var)[0, 0]

    W = vh[:r + 1].T

    return W
