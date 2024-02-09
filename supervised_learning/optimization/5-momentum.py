#!/usr/bin/env python3
"""
updates a variable using the gradient descent with
momentum optimization algorithm
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent
    with momentum optimization algorithm
    """
    m = (beta1 * v) + ((1 - beta1) * grad)
    var -= (alpha * m)
    return var, m
