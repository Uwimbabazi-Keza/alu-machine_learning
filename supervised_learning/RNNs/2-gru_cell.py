#!/usr/bin/env python3
"""class GRUCell, which represents a gated recurrent unit"""

import numpy as np


class GRUCell:
    """class GRUCell"""
    def __init__(self, i, h, o):
        """Initialize"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward propagation"""
        concatenated = np.concatenate((h_prev, x_t), axis=1)
        
        z_t = self.sigmoid(np.dot(concatenated, self.Wz) + self.bz)
        r_t = self.sigmoid(np.dot(concatenated, self.Wr) + self.br)
        
        r_h_prev = r_t * h_prev
        concatenated_r = np.concatenate((r_h_prev, x_t), axis=1)
        
        h_tilde = np.tanh(np.dot(concatenated_r, self.Wh) + self.bh)
        
        h_next = (1 - z_t) * h_prev + z_t * h_tilde
        
        y_raw = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_raw)
        
        return h_next, y

    @staticmethod
    def sigmoid(x):
        """sigmoid"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """softmax"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
