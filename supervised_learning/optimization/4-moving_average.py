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
    bias_correction = 1.0

    for i, value in enumerate(data, start=1):
        weighted_sum = beta * weighted_sum + (1 - beta) * value
        corrected_weighted_sum = weighted_sum / bias_correction
        moving_avg.append(corrected_weighted_sum)

        bias_correction *= beta

    return moving_avg
