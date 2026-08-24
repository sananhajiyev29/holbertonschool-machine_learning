#!/usr/bin/env python3
"""Module that computes the policy with a weight of a matrix."""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with a weight of a matrix.

    Args:
        matrix: the state matrix.
        weight: the weight matrix.

    Returns:
        The policy as a numpy.ndarray of action probabilities.
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z))

    return exp / np.sum(exp)
