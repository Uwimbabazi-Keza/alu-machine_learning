#!/usr/bin/env python3
"""a class NeuralNetwork that defines a neural network with one hidden layer
performing binary classification (based on 9-neural_network.py):"""


import numpy as np


class NeuralNetwork:
    """a class NeuralNetwork that defines a neural network
    with one hidden layer performing binary classification (based on 9-neural_network.py):"""

    def __init__(self, nx, nodes):
        """initialize"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """weights"""
        return (self.__W1)

    @property
    def b1(self):
        """bias"""
        return (self.__b1)

    @property
    def A1(self):
        """activated output"""
        return (self.__A1)

    @property
    def W2(self):
        """weights for the output neuron"""
        return (self.__W2)

    @property
    def b2(self):
        """bias for the output neuron"""
        return (self.__b2)

    @property
    def A2(self):
        """activated output for the output neuron"""
        return (self.__A2)

    def forward_prop(self, X):
        """forward propagation"""
        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + (np.exp(-z1)))
        z2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + (np.exp(-z2)))
        return (self.A1, self.A2)
