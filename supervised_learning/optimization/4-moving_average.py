#!/usr/bin/env python3
"""
calculates the weighted moving average
of a data set
"""


def moving_average(data, beta):
    """
    Calculate the weighted moving average of a
    data set with bias correction.
    """
    moving_avg = []
    weighted_sum = 0.0
    correction_factor = 1.0

    for i, value in enumerate(data):
        weighted_sum = beta * weighted_sum + (1 - beta) * value
        avg = weighted_sum / correction_factor
        moving_avg.append(avg)

        correction_factor *= beta

    return moving_avg
