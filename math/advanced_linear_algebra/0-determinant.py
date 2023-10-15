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
    if num_rows == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if len(row) == 0 and num_rows == 1:
            return 1
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


if __name__ == '__main__':

    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(determinant(mat0))
    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat4))
    try:
        determinant(mat5)
    except Exception as e:
        print(e)
    try:
        determinant(mat6)
    except Exception as e:
        print(e)