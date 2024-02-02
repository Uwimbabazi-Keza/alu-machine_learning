#!/usr/bin/env python3
"""updates the weights and biases of a
neural network using gradient descent with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a
    neural network using gradient descent with L2 regularization
    """

    m = Y.shape[1]

    dZ = cache["A" + str(L)] - Y
    dW = (1 / m) * np.dot(dZ, cache["A" + str(L - 1)].T) + (lambtha / m) * weights["W" + str(L)]
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    weights["W" + str(L)] -= alpha * dW
    weights["b" + str(L)] -= alpha * db

    for l in range(L, 0, -1): 
        dZ = np.dot(weights["W" + str(l + 1)].T, dZ) * (
            1 - np.power(cache["A" + str(l)], 2))
         
        dW = (1 / m) * np.dot(dZ, cache["A" + str(l - 1)].T) + (
            lambtha / m) * weights["W" + str(l)]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db

        
