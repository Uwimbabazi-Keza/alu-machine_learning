#!/usr/bin/env python3
"""
a function def matrix_transpose(matrix)
"""


def matrix_transpose(matrix):
    """
    returns the transpose of a 2D matrix
    """
    matrix_transpose = []
    for i in range(len(matrix[0])):
        matrix_transpose.append([])
    for row in matrix:
        for i in range(len(row)):
            matrix_transpose[i].append(row[i])
    return matrix_transpose
