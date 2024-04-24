#!/usr/bin/env python3
"""a class Neuron that defines a single neuron
performing binary classification (Based on 4-neuron.py)"""


import numpy as np


class Neuron:
    """a class Neuron that defines a single neuron
    performing binary classification (Based on 4-neuron.py)"""

    def __init__(self, nx):
        """intialize"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive')

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """weights"""
        return self.__W

    @property
    def b(self):
        """bias"""
        return self.__b

    @property
    def A(self):
        """Activated output"""
        return self.__A

    def forward_prop(self, X):
        """forward propagation"""
        z = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """cost"""
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        pred = self.forward_prop(X)
        cost = self.cost(Y, pred)
        pred = np.where(pred > 0.5, 1, 0)
        return (pred, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculate one pass of gradient descent on the neuron"""
        dz = A - Y
        m = X.shape[1]
        dw = (1/m) * np.matmul(dz, X.T)
        db = np.mean(dz)
        self.__W -= alpha * dw
        self.__b -= alpha * db
