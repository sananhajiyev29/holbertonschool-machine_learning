#!/usr/bin/env python3
"""
Module that defines a function to multiply two 2D matrices.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication between two 2D matrices.

    Args:
        mat1 (list of list of int/float): First matrix
        mat2 (list of list of int/float): Second matrix

    Returns:
        list: New matrix resulting from mat1 * mat2, or None if multiplication
              is not possible due to incompatible dimensions.
    """
    # Check if multiplication is possible
    if not mat1 or not mat2 or len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            # Compute the dot product of row i from mat1 and column j from mat2
            s = sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
            row.append(s)
        result.append(row)
    return result
