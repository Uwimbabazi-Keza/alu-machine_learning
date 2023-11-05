#!/usr/bin/env python3
"""a function def likelihood(x, n, P): that calculates the likelihood of
obtaining this data given various hypothetical probabilities of developing
severe side effects"""

import numpy as np


def likelihood(x, n, P):
    """function likelihood"""

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    for p in P:
        if p < 0 or p > 1:
            raise ValueError('All values in P must be in the range [0, 1]')
    fact = np.math.factorial
    f = fact(n) / (fact(n - x) * fact(x))
    likelihood = f * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
