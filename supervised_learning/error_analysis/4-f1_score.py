#!/usr/bin/env python3
""" calculates the specificity for each
class in a confusion matrix"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision

def f1_score(confusion):
    """ calculates the F1 score of a confusion matrix"""
    classes = confusion.shape[0]
    f1_per_class = np.zeros(classes)

    for i in range(classes):
        recall = sensitivity(confusion, i)
        prec = precision(confusion, i)

        f1_per_class[i] = 2 * (prec * recall) / (prec + recall) if (prec + recall) != 0 else 0

    return f1_per_class
