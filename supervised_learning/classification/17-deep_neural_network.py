#!/usr/bin/env python3
"""class DeepNeuralNetwork that defines a deep
neural network performing binary classification
(based on 16-deep_neural_network.py):"""
import numpy as np


class DeepNeuralNetwork:
    """class DeepNeuralNetwork that defines a deep neural
    network performing binary classification
    (based on 16-deep_neural_network.py):"""

    def __init__(self, nx, layers):
        """initialize"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if (
            type(layers) is not list
            or len(layers) < 1
            or min(layers) < 1
        ):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}

        self.__weights = {}
        for layer in range(self.__L):
            if layer == 0:
                self.__weights['W1'] = np.random.randn(
                    layers[0], nx) * np.sqrt(2 / nx)
                self.__weights['b1'] = np.zeros([layers[0], 1])

            else:
                self.__weights['W{}'.format(layer+1)] = np.random.randn(
                    layers[layer],
                    layers[layer-1]) * np.sqrt(2. / layers[layer-1])

                self.__weights['b{}'.format(
                    layer+1)] = np.zeros((layers[layer], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
