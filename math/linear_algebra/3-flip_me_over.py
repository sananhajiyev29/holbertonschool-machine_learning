#!/usr/bin/env python3
"""
Module that defines a function to transpose a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix

    Args:
        matrix (list): A 2D list representing the matrix

    Returns:
        list: A new matrix which is the transpose of matrix
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
