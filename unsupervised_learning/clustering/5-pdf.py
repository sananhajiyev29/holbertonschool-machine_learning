#!/usr/bin/env python3
"""Module for calculating a Gaussian probability density function."""
import numpy as np


def pdf(X, m, S):
    """Calculate the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing data points.
        m: numpy.ndarray of shape (d,) containing the mean.
        S: numpy.ndarray of shape (d, d) containing the covariance.

    Returns:
        A numpy.ndarray of shape (n,) containing PDF values,
        or None on failure.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(m, np.ndarray) or len(m.shape) != 1 or
            not isinstance(S, np.ndarray) or len(S.shape) != 2):
        return None
    if X.shape[1] != m.shape[0] or S.shape != (m.shape[0], m.shape[0]):
        return None

    d = X.shape[1]
    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    if det <= 0:
        return None

    diff = X - m
    exponent = -0.5 * np.sum(np.matmul(diff, inv) * diff, axis=1)
    denominator = np.sqrt(((2 * np.pi) ** d) * det)
    P = np.exp(exponent) / denominator

    return np.maximum(P, 1e-300)
