#!/usr/bin/env python3
"""shuffles the data points in two
matrices the same way
"""

import numpy as np

def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices
    the same way.
    """
    shuffled_indices = np.random.permutation(X.shape[0])
    shuffled_X = X[shuffled_indices, :]
    shuffled_Y = Y[shuffled_indices, :]

    return shuffled_X, shuffled_Y
