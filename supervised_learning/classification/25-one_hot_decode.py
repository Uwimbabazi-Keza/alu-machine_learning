#!/usr/bin/env python3
"""a function def one_hot_decode(one_hot):
that converts a one-hot matrix into a vector of labels:"""
import numpy as np


def one_hot_decode(one_hot):
    """one-hot"""

    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    numeric_labels = np.argmax(one_hot, axis=0)
    return numeric_labels
