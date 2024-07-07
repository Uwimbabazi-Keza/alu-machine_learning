#!/usr/bin/env python3
"""Performs the forward algorithm for a hidden markov model"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden Markov model"""
    if not isinstance(
        Observation, np.ndarray) or not isinstance(
            Emission, np.ndarray) or not isinstance(
                Transition, np.ndarray) or not isinstance(
                    Initial, np.ndarray):
        return None, None
    if len(
        Observation.shape) != 1 or len(
            Emission.shape) != 2 or len(
                Transition.shape) != 2 or len(
                    Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[
        0] != N or Transition.shape[
            1] != N or Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    F = np.zeros((N, T))

    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(
                F[:, t - 1] * Transition[:, j] * Emission[j, Observation[t]])

    P = np.sum(F[:, -1])

    return P, F
