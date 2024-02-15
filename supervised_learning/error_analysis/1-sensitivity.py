#!/usr/bin/env python3
"""calculates the sensitivity for each
class in a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each
    class in a confusion matrix"""
    
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    
    sensitivity = np.where(true_positives + false_negatives == 0, 0, true_positives / (true_positives + false_negatives))
    
    return sensitivity
