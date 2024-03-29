#!/usr/bin/env python3
"""calculates the precision for each class
in a confusion matri:"""

import numpy as np


def precision(confusion):
    """calculates the precision for each class
    in a confusion matrix"""
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    precision = np.where(true_positives + false_positives == 0, 0,
                         true_positives / (true_positives + false_positives))
    return precision
