#!/usr/bin/env python3
"""Create the class RNNCell that represents a cell of a simple RNN:
"""
import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        """initialize"""
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward propagation"""
        concatenated = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concatenated, self.Wh) + self.bh)
        y_raw = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_raw)
        
        return h_next, y

    @staticmethod
    def softmax(x):
        """softmax"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
