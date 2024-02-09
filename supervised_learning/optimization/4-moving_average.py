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
    v = 0

    for i in range(len(data)):
        v = (beta * v) + ((1 - beta) * data[i])
        moving_avg.append(v / (1 - (beta ** (i + 1))))

    return moving_avg
