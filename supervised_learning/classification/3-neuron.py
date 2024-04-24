#!/usr/bin/env python3
"""class Neuron that defines a single neuron performing
binary classification (Based on 2-neuron.py):
"""


import numpy as np


class Neuron:
    """ class Neuron that defines a single
    neuron performing binary classification (Based on 2-neuron.py):
    """

    def __init__(self, nx):
        """ initialize
        """
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
        """activated output"""
        return self.__A

    def forward_prop(self, X):
        """forward propagation
        """
        z = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """calculate cost function
        """
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost
