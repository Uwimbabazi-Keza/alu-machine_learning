#!/usr/bin/env python3
"""Calculates the most likely sequence of
    hidden states for a hidden Markov model"""

import numpy as np

def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of
    hidden states for a hidden Markov model"""
    if not isinstance(
        Observation, np.ndarray) or not isinstance(
            Emission, np.ndarray) or not isinstance(
                Transition, np.ndarray) or not isinstance(
                    Initial, np.ndarray):
        return None, None
    if len(
        Observation.shape) != 1 or len(
            Emission.shape) != 2 or len(Transition.shape) != 2 or len(Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[
        1] != N or Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)
    
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    
    for t in range(1, T):
        for j in range(N):
            probabilities = V[:, t - 1] * Transition[:, j] * Emission[j, Observation[t]]
            V[j, t] = np.max(probabilities)
            B[j, t] = np.argmax(probabilities)
    
    P = np.max(V[:, -1])
    last_state = np.argmax(V[:, -1])
    
    path = [last_state]
    for t in range(T - 1, 0, -1):
        last_state = B[last_state, t]
        path.insert(0, last_state)
    
    return path, P
