#!/usr/bin/env python3
"""class LSTMCell that represents an LSTM unit"""

import numpy as np


class LSTMCell:
    """LSTMCELL"""
    def __init__(self, i, h, o):
        """Initialize"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """propagation"""
        concatenated = np.concatenate((h_prev, x_t), axis=1)

        f_t = self.sigmoid(np.dot(concatenated, self.Wf) + self.bf)
        u_t = self.sigmoid(np.dot(concatenated, self.Wu) + self.bu)
        c_tilde = np.tanh(np.dot(concatenated, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * c_tilde

        o_t = self.sigmoid(np.dot(concatenated, self.Wo) + self.bo)
        h_next = o_t * np.tanh(c_next)

        y_raw = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_raw)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """sigmoid"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """softmax"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
