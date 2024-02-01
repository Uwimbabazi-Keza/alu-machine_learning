#!/usr/bin/env python3
"""function that calculates the cost of a neural network
with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network
    with L2 regularization
    """

    reg_term = 0

    for l in range(1, L + 1):
        reg_term += np.sum(np.square(weights["W" + str(l)]))

    reg_term *= (lambtha / (2 * m))

    cost_with_reg = cost + reg_term

    return cost_with_reg
