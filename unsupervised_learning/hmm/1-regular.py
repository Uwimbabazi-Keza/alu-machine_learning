#!/usr/bin/env python3
"""Determines the steady state probabilities of a regular Markov chain."""

import numpy as np


def regular(P):
    """Determines the steady
    state probabilities of a regular Markov chain."""
    if not isinstance(P, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    
    n = P.shape[0]
    
    A = P.T - np.eye(n)
    
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1
    
    try:
        steady_state = np.linalg.lstsq(A, b, rcond=None)[0]
        return steady_state
    except np.linalg.LinAlgError:
        return None
