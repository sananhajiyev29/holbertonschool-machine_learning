#!/usr/bin/env python3
"""
Module that slices a numpy matrix along specific axes.
"""
import numpy as np


def np_slice(matrix, axes={}):
    """
    Slices a numpy.ndarray along specified axes.
    """
    slices = [slice(None)] * matrix.ndim
    for axis, params in axes.items():
        slices[axis] = slice(*params)
    return matrix[tuple(slices)]
