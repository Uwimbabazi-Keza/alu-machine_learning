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
    