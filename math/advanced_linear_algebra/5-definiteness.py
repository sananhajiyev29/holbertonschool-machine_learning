#!/usr/bin/env python3
"""Module for determining the definiteness of a matrix."""

import numpy as np


def definiteness(matrix):
    """Determines the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): matrix to evaluate

    Returns:
        str: definiteness type or None

    Raises:
        TypeError: if matrix is not a numpy.ndarray
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.size == 0:
        return None

    # Matrix must be symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None
