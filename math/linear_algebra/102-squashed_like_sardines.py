#!/usr/bin/env python3
"""
Module that concatenates two matrices along a specified axis.
"""

def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a given axis.
    Returns a new matrix if shapes are compatible, else None.
    """
    # If both are not lists (elements), cannot go deeper for axis>0
    if not isinstance(mat1, list) and not isinstance(mat2, list):
        return [mat1, mat2] if axis == 0 else None

    # If only one is a list, shapes mismatch
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    # Axis 0: concatenate at this level
    if axis == 0:
        return mat1 + mat2

    # Axis > 0: dimensions must match
    if len(mat1) != len(mat2):
        return None

    result = []
    for sub1, sub2 in zip(mat1, mat2):
        combined = cat_matrices(sub1, sub2, axis=axis - 1)
        if combined is None:
            return None
        result.append(combined)
    return result
