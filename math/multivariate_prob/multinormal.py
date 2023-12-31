#!/usr/bin/env python3
"""
class MultiNormal that represents a
Multivariate Normal distribution
"""

import numpy as np


class MultiNormal:
    """
    class MultiNormal
    """
    def __init__(self, data):
        """
        intialize
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        centered_data = data - self.mean
        self.cov = np.dot(centered_data, centered_data.T) / (n - 1)

    def pdf(self, x):
        """
        def pdf
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d, _ = self.cov.shape
        if x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt((2 * np.pi) ** d * det)
        diff = x - self.mean
        exponent = -0.5 * diff.T @ inv @ diff
        pdf *= np.exp(exponent[0, 0])
        return pdf
