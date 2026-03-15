#!/usr/bin/env python3
"""Module that normalizes a matrix."""

import numpy as np


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix.

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize.
        m: numpy.ndarray of shape (nx,) with mean of all features of X.
        s: numpy.ndarray of shape (nx,) with std of all features of X.

    Returns:
        The normalized X matrix.
    """
    return (X - m) / s
