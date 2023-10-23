#!/usr/bin/env python3
"""
a function that calculates a sum
"""


def summation_i_squared(n):
    """
    def summation_i_squared(n)
    """
    if isinstance(n, int) and n >= 1:
        # Using the formula for the sum of squared values
        result = (n * (n + 1) * ((2 * n) + 1)) / 6
        return result
    else:
        return None
