#!/usr/bin/env python3
"""a class Deep NeuralNetwork that defines a deep neural
network performing binary classification:"""
import numpy as np


class DeepNeuralNetwork:
    """a class DeepNeuralNetwork that defines a
    deep neural network performing binary classification:"""

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

        self.L = len(layers)
        self.cache = {}

        self.weights = {}
        for layer in range(self.L):
            if layer == 0:
                self.weights['W1'] = np.random.randn(
                    layers[0], nx) * np.sqrt(2 / nx)
                self.weights['b1'] = np.zeros([layers[0], 1])

            else:
                self.weights['W{}'.format(layer+1)] = np.random.randn(
                    layers[layer],
                    layers[layer-1]) * np.sqrt(2. / layers[layer-1])

                self.weights['b{}'.format(
                    layer+1)] = np.zeros((layers[layer], 1))
