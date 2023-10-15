#!/usr/bin/env python3
"""
a function def determinant(matrix): that calculates the determinant of a matrix
"""


def determinant(matrix):
    """
    def determinant(matrix)
    """

    if not isinstance(matrix, list) or not \
            all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    num_rows = len(matrix)
    if num_rows is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if len(row) != num_rows:
            raise ValueError("matrix must be a square matrix")

    if num_rows == 1:
        return matrix[0][0]

    if num_rows == 2:
        a, b = matrix[0][0], matrix[0][1]
        c, d = matrix[1][0], matrix[1][1]
        return a * d - b * c

    d = 0
    for col in range(num_rows):
        minor_matrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        cofactor = matrix[0][col] * determinant(minor_matrix)
        d += cofactor if col % 2 == 0 else -cofactor

    return d
