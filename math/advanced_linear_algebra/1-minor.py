#!/usr/bin/env python3

"""
a function def minor(matrix): that calculates the minor matrix of a matrix
"""

determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    def minor(matrix)
    """
    if not isinstance(matrix, list) or \
            not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    num_rows = len(matrix)
    for row in matrix:
        if len(row) != num_rows:
            raise ValueError('matrix must be a non-empty square matrix')

    if num_rows == 1:
        return [[1]]
    elif num_rows == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    else:
        minor_matrix = []
        for i in range(num_rows):
            minor_row = []
            for j in range(num_rows):
                minors = [row[:j] + row[j + 1:] for \
                          row in (matrix[:i] + matrix[i + 1:])]
                minor_row.append(determinant(minors))
            minor_matrix.append(minor_row)

        return minor_matrix
