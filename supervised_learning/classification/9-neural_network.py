#!/usr/bin/env python3
"""a class NeuralNetwork that defines a neural
network with one hidden layer performing binary classification (based on 8-neural_network.py):
"""

import numpy as np


class NeuralNetwork:
    """a class NeuralNetwork that defines a neural network with one hidden
    layer performing binary classification (based on 8-neural_network.py):
    """

    def __init__(self, nx, nodes):
        """ initialize"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """weights"""
        return self.__W1

    @property
    def b1(self):
        """bias"""
        return self.__b1

    @property
    def A1(self):
        """activated output"""
        return self.__A1

    @property
    def W2(self):
        """weights for output neuron"""
        return self.__W2

    @property
    def b2(self):
        """bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """activated output for output neuron"""
        return self.__A2
