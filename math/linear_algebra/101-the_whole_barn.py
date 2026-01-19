#!/usr/bin/env python3
"""
Module that adds two matrices of any shape recursively.
"""


def add_matrices(mat1, mat2):
    """
    Recursively adds two matrices (lists) of the same shape.
    Returns a new matrix if shapes match, else None.
    """
    # Check if types match
    if type(mat1) != type(mat2):
        return None

    # If lists, check lengths and recurse
    if isinstance(mat1, list):
        if len(mat1) != len(mat2):
            return None
        res = []
        for a, b in zip(mat1, mat2):
            added = add_matrices(a, b)
            if added is None:  # stop immediately on mismatch
                return None
            res.append(added)
        return res

    # Base case: elements are numbers
    return mat1 + mat2
