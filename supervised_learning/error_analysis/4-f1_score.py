#!/usr/bin/env python3
""" calculates the specificity for each
class in a confusion matrix"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ calculates the F1 score of a confusion matrix"""
    precise = precision(confusion)
    sensitive = sensitivity(confusion)
    F1_score = (2 * precise * sensitive) / (precise + sensitive)

    return F1_score
