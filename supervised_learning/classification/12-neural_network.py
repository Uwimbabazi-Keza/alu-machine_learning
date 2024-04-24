#!/usr/bin/env python3
"""class NeuralNetwork that defines a neural network with one
hidden layer performing binary classification
(based on 11-neural_network.py):"""

import numpy as np


class NeuralNetwork:
    """class NeuralNetwork that defines a neural network
    with one hidden layer performing binary classification
    (based on 11-neural_network.py):"""

    def __init__(self, nx, nodes):
        """initialize"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = np.zeros((1, 1))
        self.__A2 = 0
        self.__A1 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """calculate forward propagation"""
        m = X.shape[1]

        weighted_sum_1 = np.dot(self.__W1, X) + self.__b1

        self.__A1 = 1 / (1 + np.exp(-weighted_sum_1))

        weighted_sum_2 = np.dot(self.__W2, self.__A1) + self.__b2

        self.__A2 = 1/(1 + np.exp(-weighted_sum_2)).reshape(1, -1)

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """calculates costt"""
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """evaluating the network's predictions"""
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = (A2 >= 0.5).astype(int)

        return predictions, cost