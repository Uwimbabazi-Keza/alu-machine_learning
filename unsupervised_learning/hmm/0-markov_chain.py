#!/usr/bin/env python3
"""determines the probability of a markov chain being
in a particular state after a specified number of iterations"""

import numpy as np


def markov_chain(P, s, t=1):
    """Determines the probability of a Markov chain being in a particular
    state after a specified number of iterations"""
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if P.shape[0] != s.shape[0]:
        return None
    if t < 1 or not isinstance(t, int):
        return None
    
    n = P.shape[0]
    
    s_t = np.copy(s)
    for _ in range(t):
        s_t = np.dot(s_t, P)
    
    return s_t
