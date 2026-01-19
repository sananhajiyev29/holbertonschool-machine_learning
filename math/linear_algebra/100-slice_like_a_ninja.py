#!/usr/bin/env python3
"""
Module that slices a matrix along specific axes.
"""


def np_slice(matrix, axes={}):
    """
    Slices a numpy.ndarray along specified axes.
    """
    slices = [slice(None)] * matrix.ndim
    for axis, params in axes.items():
        slices[axis] = slice(*params)
    return matrix[tuple(slices)]
