#!/usr/bin/env python3
"""normalizes (standardizes) a matrix
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalize (standardize) a matrix using mean
    and standard deviation.
    """
    normalized_X = (X - m) / s

    return normalized_X

