#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """conducts forward propagation using Dropout"""
    m = Y.shape[1]
    for layer_index in range(L, 0, -1):
        A = cache['A' + str(layer_index)]
        A_prev = cache['A' + str(layer_index - 1)]
        W = weights['W' + str(layer_index)]
        b = weights['b' + str(layer_index)]
        if layer_index == L:
            dZ = A - Y
        else:
            dZ = dA * (1 - (A ** 2))
            dZ = np.multiply(dZ, cache['D' + str(layer_index)])
            dZ /= keep_prob
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(W.T, dZ)
        weights['W' + str(layer_index)] = W - (alpha * dW)
        weights['b' + str(layer_index)] = b - (alpha * db)
    return weights
