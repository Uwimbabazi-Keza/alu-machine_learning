#!/usr/bin/env python3
""" calculates the specificity for each
class in a confusion matrix"""

import numpy as np


def specificity(confusion):
    """ calculates the specificity for each
    class in a confusion matrix"""
    
    classes = confusion.shape[0]
    spec_per_class = np.zeros(classes)

    for i in range(classes):
        true_negatives = np.sum(np.delete(confusion, i, axis=0)[:, np.delete(confusion, i, axis=1)].flatten())
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]

        spec_per_class[i] = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0

    return spec_per_class
