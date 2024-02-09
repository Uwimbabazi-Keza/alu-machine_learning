#!/usr/bin/env python3
"""calculates the normalization
(standardization) constants of a matrix
"""

import numpy as np

def normalization_constants(X):
    """
    Calculate the mean and standard deviation of
    each feature for normalization.
    """

    mean_values = np.mean(X, axis=0)
    std_dev_values = np.std(X, axis=0)

    return mean_values, std_dev_values
