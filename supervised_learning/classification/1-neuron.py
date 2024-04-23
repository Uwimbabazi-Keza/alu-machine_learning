#!/usr/bin/env python3
"""class Neuron that defines a single neuron performing binary
classification (Based on 0-neuron.py):"""


import numpy as np


class Neuron:
    """class Neuron that defines a single neuron
    performing binary classification (Based on 0-neuron.py):"""

    def __init__(self, nx):
        """class Neuron that defines a single neuron
    performing binary classification (Based on 0-neuron.py):"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """weights"""
        return (self.__W)

    @property
    def b(self):
        """bias for the neuron
        """
        return (self.__b)

    @property
    def A(self):
        """__A is the activated output of the neuron
        """
        return (self.__A)
