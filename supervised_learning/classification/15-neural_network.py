#!/usr/bin/env python3
"""class NeuralNetwork that defines a neural
network with one hidden layer performing binary
classification (based on 14-neural_network.py):"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """class NeuralNetwork that defines a neural network
    with one hidden layer performing binary classification
    (based on 14-neural_network.py):"""

    def __init__(self, nx, nodes):
        """initiate"""

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
        """calculates cost"""
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """evaluating the network's predictions"""
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = (A2 >= 0.5).astype(int)

        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculate one pass of gradient descent on the neural network"""
        m = X.shape[1]

        dA1 = np.dot(self.__W2.T, (A2 - Y))
        dZ1 = dA1 * A1 * (1 - A1)
        dw_1 = (1 / m) * np.dot(dZ1, X.T)
        db_1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        db_2 = np.sum(A2 - Y) / m
        dw_2 = np.dot((A2 - Y), A1.T) / m

        self.__W1 = self.__W1 - alpha * dw_1
        self.__W2 = self.__W2 - alpha * dw_2
        self.__b1 = self.__b1 - alpha * db_1
        self.__b2 = self.__b2 - alpha * db_2


    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Training the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0.0:
            raise ValueError("alpha must be positive")

        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        plotting_costs = []
        if graph:
            plotting_steps = np.arange(0, iterations + 1, step)
            if iterations % step != 0:
                plotting_steps = np.append(plotting_steps, iterations)

            for iteration in range(iterations):
                A1, A2 = self.forward_prop(X)
                self.gradient_descent(X, Y, A1, A2, alpha)
                if (iteration % step) == 0 or iteration == (iterations - 1):
                    cost = self.cost(Y, A2)
                    plotting_costs.append(cost)

                    if verbose:
                        print(f"Cost after {iteration} iterations: {cost}")

            plt.plot(plotting_steps, plotting_costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()
