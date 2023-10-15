#!/usr/bin/env python3

"""
a function def add_matrices2D(mat1, mat2): that adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    a function def add_matrices2D(mat1, mat2)
    """
    result_matrix = []
    if len(mat1) == len(mat2):
        for arr1, arr2 in zip(mat1, mat2):
            if len(arr1) == len(arr2):
                result_matrix.append([x + y for x, y in zip(arr1, arr2)])
            else:
                return None
        return result_matrix
