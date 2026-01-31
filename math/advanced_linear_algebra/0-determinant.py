#!/usr/bin/env python3
"""Module for calculating the determinant of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): matrix to calculate determinant of

    Returns:
        determinant of the matrix

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is not square
    """
    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        sub = []
        for row in matrix[1:]:
            sub.append(row[:col] + row[col + 1:])
        det += ((-1) ** col) * matrix[0][col] * determinant(sub)

    return det
