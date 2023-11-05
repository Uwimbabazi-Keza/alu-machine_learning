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
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")
    
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not all(0 < p < 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")
    
    likelihoods = [np.math.comb(n, x) * p**x * (1-p)**(n-x) for p in P]

    return np.array(likelihoods)
