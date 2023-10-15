#!/usr/bin/env python3
"""
a function def np_elementwise(mat1, mat2):
that performs element-wise addition, subtraction,
multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """
    a function def np_elementwise(mat1, mat2)
    """
    new = []
    new.append(mat1 + mat2)
    new.append(mat1 - mat2)
    new.append(mat1 * mat2)
    new.append(mat1 / mat2)
    return tuple(new)
