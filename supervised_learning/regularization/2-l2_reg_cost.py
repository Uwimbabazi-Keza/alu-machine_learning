#!/usr/bin/env python3
"""function that  calculates the cost of a neural
network with L2 regularization"""

import tensorflow as tf


def l2_reg_cost(cost, lambtha, weights):
    """ calculates the cost of a neural
    network with L2 regularization
    """

    l2_reg_term = sum(tf.nn.l2_loss(w) for w in weights)

    l2_reg_term *= (lambtha / 2)

    cost_with_reg = cost + l2_reg_term

    return cost_with_reg
