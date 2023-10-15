#!/usr/bin/env python3
"""
a function def mat_mul(mat1, mat2):
that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    def mat_mul(mat1, mat2)
    """
    if len(mat1[0]) != len(mat2):
        return None
    elif len(mat1[0]) == len(mat2):
        return [[sum(a * b for a, b in zip(x, y))
                 for y in zip(*mat2)]
                for x in mat1]
