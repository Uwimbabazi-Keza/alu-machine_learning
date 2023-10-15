#!/usr/bin/env python3

"""
def adjugate(matrix): that calculates the adjugate matrix of a matrix
"""

cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    adjugate(matrix)
    """

    if not isinstance(matrix, list) or not \
            all(isinstance(row, list) for row in matrix):
        raise TypeError('Input must be a list of lists')

    num_rows = len(matrix)
    for row in matrix:
        if len(row) != num_rows:
            raise ValueError('Input must be a non-empty square matrix')

    if num_rows == 1:
        return [[1]]
    elif num_rows == 2:
        return [[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]]

    else:
        cofactor_matrix = cofactor(matrix)

        adjugate_matrix = [[cofactor_matrix[j][i] for j in range(num_rows)] for i in range(num_rows)]

        return adjugate_matrix
