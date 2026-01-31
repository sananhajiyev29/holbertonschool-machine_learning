#!/usr/bin/env python3
"""Module for calculating the adjugate matrix of a matrix."""


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix.

    Args:
        matrix (list of lists): matrix to calculate adjugate of

    Returns:
        list of lists: adjugate matrix

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is not a non-empty square matrix
    """
    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    def determinant(mat):
        """Recursive determinant helper (no imports allowed)."""
        size = len(mat)
        if size == 1:
            return mat[0][0]
        if size == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0
        for col in range(size):
            sub = []
            for row in mat[1:]:
                sub.append(row[:col] + row[col + 1:])
            det += ((-1) ** col) * mat[0][col] * determinant(sub)
        return det

    cof = []
    for i in range(n):
        row = []
        for j in range(n):
            sub = []
            for r in range(n):
                if r != i:
                    sub.append(matrix[r][:j] + matrix[r][j + 1:])
            minor_ij = determinant(sub)
            row.append(((-1) ** (i + j)) * minor_ij)
        cof.append(row)

    adj = []
    for j in range(n):
        adj_row = []
        for i in range(n):
            adj_row.append(cof[i][j])
        adj.append(adj_row)

    return adj
