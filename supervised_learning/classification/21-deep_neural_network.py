#!/usr/bin/env python3
"""class DeepNeuralNetwork that defines a deep neural network
performing binary classification (based on 20-deep_neural_network.py):"""
import numpy as np


class DeepNeuralNetwork:
    """class DeepNeuralNetwork that defines a deep neural network
    performing binary classification (based on 20-deep_neural_network.py):"""

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

    def forward_prop(self, X):
        """calculate forward propagation"""
        self.__cache['A0'] = X

        for l in range(1, self.__L+1):
            W = self.__weights['W{}'.format(l)]
            b = self.__weights['b{}'.format(l)]
            A_prev = self.__cache['A{}'.format(l-1)]

            Z = np.dot(W, A_prev) + b
            self.__cache['A{}'.format(l)] = 1 / (1 + np.exp(-Z))

        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """calculate cost of model using logistic regression"""
        m = Y.shape[1]
        cost = -(1/m) * np.sum([Y * np.log(A) +
                                (1 - Y) * np.log(1.0000001 - A)])
        return cost

    def evaluate(self, X, Y):
        """evaluate the neural network's prediction"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = (A >= 0.5).astype(int)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """calculate one pass of gradient descent on neural network
        alpha - learning rate"""
        m = Y.shape[1]
        A = self.__cache['A{}'.format(self.__L)]
        dz = A - Y

        for l in reversed(range(1, self.__L + 1)):
            dw = np.matmul(cache["A{}".format(l - 1)], dz.T) / m
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            da = cache["A{}".format(l - 1)] * (1 - cache["A{}".format(l - 1)])
            dz = np.matmul(self.__weights["W{}".format(l)].T, dz) * da
            self.__weights["W{}".format(l)] -= alpha * dw.T
            self.__weights["b{}".format(l)] -= alpha * db

        return self.__weights
