#!/usr/bin/env python3
"""a function def one_hot_encode(Y, classes):
that converts a numeric label vector into a one-hot matrix:"""
import numpy as np


def one_hot_encode(Y, classes):
    """one hot code"""

    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    m = Y.shape[0]
    one_hot_matrix = np.zeros((classes, m))

    for idx, val in enumerate(Y):
        if val >= classes or val < 0:
            return None
        one_hot_matrix[val, idx] = 1

    return one_hot_matrix
