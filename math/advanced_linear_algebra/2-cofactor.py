#!/usr/bin/env python3

"""
def cofactor(matrix): that calculates the cofactor matrix of a matrix
"""

minor = __import__('1-minor').minor


def cofactor(matrix):
    """
    def cofactor(matrix)
    """

    if not isinstance(matrix, list) or not \
            all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    num_rows = len(matrix)
    for row in matrix:
        if len(row) != num_rows:
            raise ValueError('matrix must be a non-empty square matrix')

    if num_rows == 1:
        return [[1]]
    elif num_rows == 2:
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]

    else:
        minor_matrix = minor(matrix)
        c_mat = []

        for i in range(num_rows):
            c_row = []
            for j in range(num_rows):

                sign = (-1) ** (i + j)
                c_element = sign * minor_matrix[i][j]
                c_row.append(c_element)
            c_mat.append(c_row)

        return c_mat
