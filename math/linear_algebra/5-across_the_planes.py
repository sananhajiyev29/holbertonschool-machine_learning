#!/usr/bin/env python3
"""
Module that defines a function to add two 2D matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise

    Args:
        mat1 (list of list): First 2D matrix of numbers
        mat2 (list of list): Second 2D matrix of numbers

    Returns:
        list of list: A new 2D matrix with element-wise sums
        None: If the two matrices are not the same shape
    """
    if len(mat1) != len(mat2):
        return None
    if any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None

    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
