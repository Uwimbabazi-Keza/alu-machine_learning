#!/usr/bin/env python3

"""
a function def minor(matrix): that calculates the minor matrix of a matrix
"""

determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    def minor(matrix)
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    num_rows = len(matrix)
    if num_rows == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != num_rows:
            raise ValueError("matrix must be a non-empty square matrix")

    minors = []

    for i in range(num_rows):
        minor_row = []
        for j in range(num_rows):
            minor_matrix = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
            minor_det = determinant(minor_matrix)
            minor_row.append(minor_det)
        minors.append(minor_row)

    return minors
