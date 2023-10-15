#!/usr/bin/env python3
"""a function def np_cat(mat1, mat2, axis=0)
that concatenates two matrices along a specific axis
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    def np_cat(mat1, mat2, axis=0)
    """
    return np.concatenate((mat1, mat2), axis=axis)
