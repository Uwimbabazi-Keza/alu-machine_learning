#!/usr/bin/env python3
"""Performs the backward algorithm for a hidden Markov model"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden Markov model"""
    if not isinstance(
        Observation, np.ndarray) or not isinstance(
            Emission, np.ndarray) or not isinstance(
                Transition, np.ndarray) or not isinstance(Initial, np.ndarray):
        return None, None
    if len(
        Observation.shape) != 1 or len(
            Emission.shape) != 2 or len(Transition.shape) != 2 or len(
                Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if (Transition.shape[0] != N or Transition.shape[1] != N or
        Initial.shape[0] != N or Initial.shape[1] != 1):
        return None, None

    B = np.zeros((N, T))

    B[:, T-1] = 1

    for t in range(T-2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                B[:, t + 1] * Transition[i, :] * Emission[:, Observation[t + 1]])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
