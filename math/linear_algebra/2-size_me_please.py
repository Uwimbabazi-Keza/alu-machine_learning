#!/usr/bin/env python3
"""
a function def matrix_shape(matrix): that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """
    a function def matrix_shape(matrix)
    """
    shape = []
    while (type(matrix) is list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
