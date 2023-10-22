#!/usr/bin/env python3
"""
 class MultiNormal that represents a
 Multivariate Normal distribution"""

import numpy as np


class MultiNormal:
    """
    class MultiNormal
    """
    def __init__(self, data):
        """
        intialize
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = (data - self.mean) @ (data - self.mean).T / (n - 1)

    def pdf(self, x):
        if not isinstance(x, np.ndarray) or x.shape != (self.cov.shape[0], 1):
            raise ValueError(f"x must be a numpy.ndarray with shape ({self.cov.shape[0]}, 1)")
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt((2 * np.pi) ** self.cov.shape[0] * det)
        diff = x - self.mean
        exponent = -0.5 * diff.T @ inv @ diff
        pdf *= np.exp(exponent[0, 0])
        return pdf
