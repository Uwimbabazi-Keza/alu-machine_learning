#!/usr/bin/env python3
"""performs forward propagation for a simple RNN"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """rnn function"""
    t, m, i = X.shape
    _, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    outputs = []
    
    h_prev = h_0
    for step in range(t):
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        outputs.append(y)
        h_prev = h_next
    
    Y = np.array(outputs)
    return H, Y
