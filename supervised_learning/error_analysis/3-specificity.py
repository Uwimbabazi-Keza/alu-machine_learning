#!/usr/bin/env python3
""" calculates the specificity for each
class in a confusion matrix"""

import numpy as np


def specificity(confusion):
    """ calculates the specificity for each
    class in a confusion matrix"""

    classes = confusion.shape[0]
    specificity = np.zeros(classes)

    for i in range(classes):
        true_negatives = np.sum(confusion) - np.sum(
            confusion[i]) - np.sum(confusion[:, i]) + confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]

        specificity[i] = true_negatives / (
             + false_positives) if (true_negatives + false_positives) != 0 else 0

    return specificity
