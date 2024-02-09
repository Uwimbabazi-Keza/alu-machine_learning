#!/usr/bin/env python3

"""
normalizes an unactivated output of a neural
network using batch normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network
    using batch normalization
    """

    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    zn = (Z - mean) / ((variance + epsilon) ** 0.5)
    zn = gamma * zn + beta
    return zn
