#!/usr/bin/env python3
"""
Module that adds two matrices of any shape recursively.
"""


def add_matrices(mat1, mat2):
    """
    Recursively adds two matrices (lists) of the same shape.
    Returns a new matrix if shapes match, else None.
    """
    if type(mat1) != type(mat2):
        return None

    if isinstance(mat1, list):
        if len(mat1) != len(mat2):
            return None
        return [add_matrices(a, b) for a, b in zip(mat1, mat2)]
    return mat1 + mat2
