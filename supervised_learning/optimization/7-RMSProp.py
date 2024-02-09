#!/usr/bin/env python3
"""
updates a variable using RMSProp optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using RMSProp optimization algorithm
    """
    update = beta2 * s + (1 - beta2) * grad ** 2
    var -= alpha * grad / (epsilon + update ** 0.5)
    return var, update
