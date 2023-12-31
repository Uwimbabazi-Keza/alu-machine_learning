#!/usr/bin/env python3

"""
def mean_cov(X): that calculates the
mean and covariance of a data set
"""

import numpy as np


def mean_cov(X):
    """
    def mean_cov(X)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    centered_data = X - mean
    c = np.dot(centered_data.T, centered_data) / (n - 1)

    return mean, c
