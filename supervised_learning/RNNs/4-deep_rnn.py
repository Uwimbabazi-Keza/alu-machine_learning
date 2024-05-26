#!/usr/bin/env python3
"""performs forward propagation for a deep RNN"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """forward propagation"""
    t, m, i = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    outputs = []

    for step in range(t):
        x_t = X[step]

        for layer in range(l):
            h_prev = H[step, layer]
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            else:
                h_next, y = rnn_cells[layer].forward(h_prev, H[step, layer - 1])

            H[step + 1, layer] = h_next

        outputs.append(y)

    Y = np.array(outputs)
    return H, Y
