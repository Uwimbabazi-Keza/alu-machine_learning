#!/usr/bin/env python3

"""
def definiteness(matrix): that calculates the definiteness of a matrix
"""

import numpy as np


def definiteness(matrix):
    """
    def definiteness(matrix)
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or not np.array_equal(matrix, matrix.T):
        return None

    positive_count = 0
    negative_count = 0
    zero_count = 0

    eigenvalues = np.linalg.eig(matrix)[0]

    for value in eigenvalues:
        if value > 0:
            positive_count += 1
        if value < 0:
            negative_count += 1
        if value == 0:
            zero_count += 1

    if positive_count and zero_count and not negative_count:
        return "Positive semi-definite"
    elif negative_count and zero_count and not positive_count:
        return "Negative semi-definite"
    elif positive_count and not negative_count:
        return "Positive definite"
    elif negative_count and not positive_count:
        return "Negative definite"

    return "Indefinite"
