#!/usr/bin/env python3
"""
1-intersection.py, write a function def marginal(x, n, P, Pr):
that calculates the marginal probability of obtaining the data
"""


import numpy as np


def marginal(x, n, P, Pr):
    """def marginal(x, n, P, Pr):"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for p in P:
        if p < 0 or p > 1:
            raise ValueError('All values in P must be in the range [0, 1]')
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    for i in Pr:
        if i < 0 or i > 1:
            raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    f = np.math.factorial
    likelihood = f(n)/(f(x)*f(n-x))
    likelihood *= (P**x) * ((1-P)**(n-x))
    return np.sum(likelihood * Pr)