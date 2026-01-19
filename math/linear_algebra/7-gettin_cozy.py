#!/usr/bin/env python3
"""
Module that defines a function to concatenate two 2D matrices
along a specified axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specified axis

    Args:
        mat1 (list of list of int/float): First 2D matrix
        mat2 (list of list of int/float): Second 2D matrix
        axis (int): Axis along which to concatenate (0 for rows,
                    1 for columns)

    Returns:
        list: New concatenated 2D matrix, or None if not possible
    """
    if axis == 0:
        # Concatenate along rows: mat2 must have same number of columns
        if not mat1 or not mat2 or len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        # Concatenate along columns: mat1 and mat2 must have same number of rows
        if len(mat1) != len(mat2):
            return None
        return [row1[:] + row2[:] for row1, row2 in zip(mat1, mat2)]
    else:
        return None
