#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout
    """
    outputs = {"A0": X}

    for layer_index in range(1, L + 1):
        weight = weights["W{}".format(layer_index)]
        bias = weights["b{}".format(layer_index)]
        Z = np.matmul(weight, outputs["A{}".format(
            layer_index - 1)]) + bias
        dropout = np.random.binomial(1, keep_prob, size=Z.shape)

        if layer_index != L:
            outputs["A{}".format(layer_index
                                 )] = np.tanh(Z) * dropout / keep_prob
            outputs["D{}".format(layer_index)] = dropout
        else:
            exp_Z = np.exp(Z)
            outputs["A{}".format(layer_index)] = exp_Z / np.sum(
                exp_Z, axis=0, keepdims=True)

    return outputs
