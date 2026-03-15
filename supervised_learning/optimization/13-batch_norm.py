#!/usr/bin/env python3
"""Module that normalizes an unactivated output using batch normalization."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output using batch normalization.

    Args:
        Z: numpy.ndarray of shape (m, n) that should be normalized.
        gamma: numpy.ndarray of shape (1, n) containing the scales.
        beta: numpy.ndarray of shape (1, n) containing the offsets.
        epsilon: small number to avoid division by zero.

    Returns:
        The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
