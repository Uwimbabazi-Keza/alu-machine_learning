#!/usr/bin/env python3
"""Update the class DeepNeuralNetwork to perform
multiclass classification
(based on 26-deep_neural_network.py):"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """Update the class DeepNeuralNetwork to
    perform multiclass classification
    (based on 26-deep_neural_network.py):"""

    def __init__(self, nx, layers):
        """intialize"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif not isinstance(layers, list) or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        weights = {}
        prev = nx
        for i, L in enumerate(layers, 1):
            if not isinstance(L, int) and L < 0:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(i)] = np.zeros((L, 1))
            weights["W{}".format(i)] = (np.random.randn(L, prev) *
                                        np.sqrt(2 / prev))
            prev = L
        self.__weights = weights

    @property
    def L(self):
        """Getter for self.__L"""
        return self.__L

    @property
    def cache(self):
        """Getter for self.__cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for self.__weights"""
        return self.__weights

    def forward_prop(self, X):
        """Function that calculates the forward propagation of the deep
            neural network."""
        A = X
        self.cache["A{}".format(0)] = X
        for L in range(1, self.L + 1):
            prev = A
            W = self.weights["W{}".format(L)]
            b = self.weights["b{}".format(L)]
            Z = np.matmul(W, A) + b
            if L == self.L:
                T = np.exp(Z)
                A = T / np.sum(T, axis=0, keepdims=True)  # Softmax
            else:
                A = 1/(1 + np.exp(-Z))  # Sigmoid
            self.__cache["A{}".format(L)] = A
        return A, self.cache

    def cost(self, Y, A):
        """Function that alculates the cost of the model using
            logistic regression."""
        m = Y.shape[1]
        loss = np.sum(-Y * np.log(A))
        return loss / m

    def evaluate(self, X, Y):
        """Function that valuates the deep neural network’s predictions."""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Function that calculates one pass of gradient descent on the neural
        network."""
        m = Y.shape[1]

        for i in range(self.L, 0, -1):
            Ai = cache["A{}".format(i)]
            Ai_next = cache["A{}".format(i - 1)]

            if i == self.L:
                dZi = (Ai - Y)
            else:
                dZi = dAi_next * (Ai * (1 - Ai))  # σ'((z)) = σ(z)(1-σ(z))
            dWi = np.matmul(dZi, Ai_next.T) / m
            dbi = np.sum(dZi, axis=1, keepdims=True) / m

            dZ_prev = dZi
            Wi = self.weights["W{}".format(i)]
            dAi_next = np.matmul(Wi.T, dZi)

            self.__weights["W{}".format(i)] -= (alpha * dWi)
            self.__weights["b{}".format(i)] -= (alpha * dbi)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Function that trains the neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        elif not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        x = np.arange(0, iterations, step)
        y = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if (i) % step == 0 or i == 0:
                cost = self.cost(Y, A)
                y.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}"
                          .format((i), cost))
        if verbose:
            print("Cost after {} iterations: {}"
                  .format((i + 1), cost))

        if graph:
            plt.plot(x, y, "b-")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Function that saves the instance
        object to a file in pickle format"""
        ext = ".pkl"
        if ext not in filename:
            filename += ext
        try:
            with open(filename, "wb") as file:
                pickle.dump(self, file)
        except Exception:
            return None

    def load(filename):
        """Function that loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as file:
                contents = pickle.load(file)
                return contents
        except Exception:
            return None
