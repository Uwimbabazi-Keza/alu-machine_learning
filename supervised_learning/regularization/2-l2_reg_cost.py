#!/usr/bin/env python3
"""function that  calculates the cost of a neural
network with L2 regularization"""

import tensorflow as tf


def l2_reg_cost(cost):
    """ calculates the cost of a neural
    network with L2 regularization
    """

    l2_reg_term =  tf.losses.get_regularization_losses()
    
    cost_with_reg = cost + l2_reg_term

    return cost_with_reg
