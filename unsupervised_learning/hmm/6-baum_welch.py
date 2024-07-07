#!/usr/bin/env python3
"""Performs the Baum-Welch algorithm for a hidden markov mode"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden Markov model"""
    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))

    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t - 1] * Transition[:, j]) * Emission[j, Observation[t]]

    P = np.sum(F[:, -1])
    return P, F

def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden Markov model"""
    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.zeros((N, T))
    
    B[:, -1] = 1
    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(B[:, t + 1] * Transition[i, :] * Emission[:, Observation[t + 1]])

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden Markov model"""
    T = Observations.shape[0]
    N = Transition.shape[0]
    M = Emission.shape[1]

    for _ in range(iterations):
        _, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        Xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            denom = np.sum(
                np.sum(F[:, t].reshape(-1, 1) * Transition * Emission[:, Observations[t + 1]].reshape(
                    1, -1) * B[:, t + 1].reshape(1, -1), axis=0), axis=0)
            for i in range(N):
                numer = F[i, t] * Transition[i, :] * Emission[:, Observations[t + 1]] * B[:, t + 1]
                Xi[i, :, t] = numer / denom

        Gamma = np.sum(Xi, axis=1)

        Transition = np.sum(Xi, 2) / np.sum(Gamma, axis=1).reshape(-1, 1)

        Gamma = np.hstack((Gamma, np.sum(Xi[:, :, T - 2], axis=0).reshape(-1, 1)))

        denominator = np.sum(Gamma, axis=1).reshape(-1, 1)
        numerator = np.zeros((N, M))

        for t in range(T):
            numerator[:, Observations[t]] += Gamma[:, t]

        Emission = numerator / denominator

        Initial = Gamma[:, 0].reshape(-1, 1)

    return Transition, Emission
