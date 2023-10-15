#!/usr/bin/env python3

"""
def inverse(matrix): that calculates the inverse of a matrix
"""

determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
    def inverse(matrix)
    """
    if not isinstance(matrix, list) or not \
            all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    num_rows = len(matrix)
    for row in matrix:
        if len(row) != num_rows:
            raise ValueError('matrix must be a non-empty square matrix')

    d = determinant(matrix)

    if num_rows == 1:
        return [[1 / matrix[0][0]]] if matrix[0][0] != 0 else None
    elif num_rows == 2:
        if d == 0:
            return None
        return [[matrix[1][1] / d, -matrix[0][1] / d],
                [-matrix[1][0] / d, matrix[0][0] / d]]
    elif d == 0:
        return None
    else:
        adjugate_matrix = adjugate(matrix)
        determinant_value = determinant(matrix)

        return [[entry / determinant_value for entry in row] 
                for row in adjugate_matrix]
