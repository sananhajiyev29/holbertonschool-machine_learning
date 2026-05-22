#!/usr/bin/env python3
"""Module that calculates Shannon entropy and P affinities for t-SNE."""
import numpy as np


def HP(Di, beta):
    """Calculates the Shannon entropy and P affinities relative to a point.

    Args:
        Di: numpy.ndarray of shape (n - 1,) containing the pairwise
            distances between a data point and all other points.
        beta: numpy.ndarray of shape (1,) containing the beta value for
            the Gaussian distribution.

    Returns:
        Tuple of (Hi, Pi) where Hi is the Shannon entropy and Pi is the
        P affinities.
    """
    num = np.exp(-Di * beta)
    sum_num = np.sum(num)
    Pi = num / sum_num
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
