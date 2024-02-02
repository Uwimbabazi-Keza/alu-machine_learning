#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """conducts forward propagation using Dropout"""

    m = Y.shape[1]

    for i in range(L, 0, -1):
        A = cache["A{}".format(i)]
        dz = A - Y

        db = np.sum(dz, axis=1, keepdims=True) / m
        dw = np.dot((cache["A{}".format(
            i - 1)] if i > 1 else cache["A0"]), dz.T) / m

        if i > 1:
            dz = np.dot(weights["W{}".format(i)].T, dz) * (1 - (
                cache["A{}".format(i - 1)] if i > 1 else cache["A0"])**2)
            dz *= cache["D{}".format(i - 1)]
            dz /= keep_prob

        weights["b{}".format(i)] -= alpha * db
        weights["W{}".format(i)] -= alpha * dw.T
