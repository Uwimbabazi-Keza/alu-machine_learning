#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""

import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout
    """
    outputs = {}
    masks = {}

    A = X
    outputs["A0"] = A

    for l in range(1, L):
        Z = np.dot(weights["W" + str(l)], A) + weights["b" + str(l)]
        A = np.tanh(Z)
        
        mask = (np.random.rand(*A.shape) < keep_prob).astype(int)
        A = A * mask / keep_prob

        outputs["Z" + str(l)] = Z
        outputs["A" + str(l)] = A
        masks["D" + str(l)] = mask

    Z_last = np.dot(weights["W" + str(L)], A) + weights["b" + str(L)]
    exp_Z_last = np.exp(Z_last)
    A_last = exp_Z_last / np.sum(exp_Z_last, axis=0, keepdims=True)

    outputs["Z" + str(L)] = Z_last
    outputs["A" + str(L)] = A_last

    return outputs, masks
