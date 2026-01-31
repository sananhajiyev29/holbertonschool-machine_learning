#!/usr/bin/env python3
"""Module for calculating the inverse of a matrix."""


def inverse(matrix):
    """Calculates the inverse of a matrix.

    Args:
        matrix (list of lists): matrix to calculate inverse of

    Returns:
        list of lists: inverse matrix, or None if matrix is singular

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

    def determinant(mat):
        """Recursive determinant helper (no imports allowed)."""
        if mat == [[]]:
            return 1

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

    def adjugate(mat):
        """Compute adjugate (transpose of cofactor matrix)."""
        size = len(mat)
        if size == 1:
            return [[1]]

        cof = []
        for i in range(size):
            row = []
            for j in range(size):
                sub = []
                for r in range(size):
                    if r != i:
                        sub.append(mat[r][:j] + mat[r][j + 1:])
                minor_ij = determinant(sub)
                row.append(((-1) ** (i + j)) * minor_ij)
            cof.append(row)

        adj = []
        for j in range(size):
            adj_row = []
            for i in range(size):
                adj_row.append(cof[i][j])
            adj.append(adj_row)
        return adj

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    inv = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adj[i][j] / det)
        inv.append(row)

    return inv
