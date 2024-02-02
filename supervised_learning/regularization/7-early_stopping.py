#!/usr/bin/env python3

"""determines if you should stop gradient descent early"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early
    """

    stop = False

    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    if count == patience:
        stop = True

    return stop, count
