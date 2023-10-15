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
        raise TypeError('matrix must be a numpy.ndarray')

    if not matrix.size or matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues = np.linalg.eigvals(matrix)
    n = len(eigenvalues)

    positive = all(e > 0 for e in eigenvalues)
    negative = all(e < 0 for e in eigenvalues)
    zero = all(e == 0 for e in eigenvalues)

    if n == 1:
        return "Positive definite" if positive else "Positive semi-definite" \
            if zero else "Negative definite"

    if n == 2:
        return "Positive definite" if positive else "Negative definite" \
            if negative else "Indefinite" if not zero else None

    if n == 3:
        return "Positive definite" if positive else "Negative definite" \
            if negative else "Indefinite" if not zero else None

    return None
