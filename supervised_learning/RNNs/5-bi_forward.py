#!/usr/bin/env python3
"""performs forward propagation for a deep RNN"""

import numpy as np


class BidirectionalCell:
    """class BidirectionalCell"""
    def __init__(self, i, h, o):
        """Initialize"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward direction"""
        concatenated = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concatenated, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """backward direction"""
        concatenated = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(concatenated, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """output of the bidirectional cell"""
        t, m, _ = H.shape
        Y = np.dot(H.reshape(t * m, -1), self.Wy) + self.by
        return Y.reshape(t, m, -1)
