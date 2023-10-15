#!/usr/bin/env python3

"""
a function def cat_matrices2D(mat1, mat2, axis=0):
that concatenates two matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    function def cat_matrices2D(mat1, mat2, axis=0)
    """
    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            return mat1 + mat2
        return None
    elif axis == 1:
        if len(mat1) == len(mat2):
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        else:
            return None
    else:
        return None
