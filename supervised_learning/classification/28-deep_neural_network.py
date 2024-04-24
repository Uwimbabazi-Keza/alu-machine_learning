#!/usr/bin/env python3

"""Update the class DeepNeuralNetwork to allow
different activation functions
(based on 27-deep_neural_network.py):"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


def _check_activation(activation: str):
    """Update the class DeepNeuralNetwork to
    allow different activation functions
    (based on 27-deep_neural_network.py):"""
    if activation != "sig" and activation != "tanh":
        raise ValueError("activation must be 'sig' or 'tanh'")


def _check_step(step: int, iterations: int) -> None:
    """
    Check the step parameter"""
    if not isinstance(step, int):
        raise TypeError("step must be an integer")
    if step <= 0 or step > iterations:
        raise ValueError("step must be positive and <= iterations")


def _check_nx(nx: int) -> None:
    """
    Check is the number of inputed features are an int and only positif"""
    if not isinstance(nx, int):
        raise TypeError("nx must be an integer")
    if nx < 1:
        raise ValueError("nx must be a positive integer")


def _check_iterations(interations: int) -> None:
    """
    Chec the iteration variable"""
    if not isinstance(interations, int):
        raise TypeError("iterations must be an integer")
    if interations <= 0:
        raise ValueError("iterations must be a positive integer")


def _check_alpha(alpha: float) -> None:
    """
    Check the learning rate parameter"""
    if not isinstance(alpha, float):
        raise TypeError("alpha must be a float")
    if alpha <= 0:
        raise ValueError("alpha must be positive")


def _check_layers(layers: int) -> None:
    """
    Check is the number of required nodes are an int and only positif"""
    if not isinstance(layers, list) or len(layers) == 0:
        raise TypeError("layers must be a list of positive integers")


class DeepNeuralNetwork:
    """Deep neural network interface"""

    def __init__(self, nx, layers, activation='sig'):
        """
        Init"""
        _check_nx(nx)
        _check_layers(layers)
        _check_activation(activation)
        self.__nx = nx
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        for layer_idx in range(self.L):
            if layers[layer_idx] <= 0:
                raise TypeError("layers must be a list of positive integers")
            current_weight_key = "W{}".format(layer_idx + 1)
            current_bias_key = "b{}".format(layer_idx + 1)
            if layer_idx == 0:
                self.weights[current_weight_key] = np.random.randn(
                    layers[layer_idx], self.nx
                ) * np.sqrt(2. / self.nx)
            else:
                self.weights[current_weight_key] = np.random.randn(
                    layers[layer_idx], layers[layer_idx - 1]
                ) * np.sqrt(2. / layers[layer_idx - 1])

            self.weights[current_bias_key] = np.zeros((layers[layer_idx], 1))

    @property
    def activation(self):
        return self.__activation

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def nx(self):
        return self.__nx

    def forward_prop(self, X):
        """Compute the forward propagation"""
        self.__cache["A0"] = X
        for layer_idx in range(self.L):
            input_key = "A{}".format(layer_idx)
            weight_key = "W{}".format(layer_idx + 1)
            bias_key = "b{}".format(layer_idx + 1)
            z = np.matmul(
                self.weights.get(weight_key),
                self.cache.get(input_key)
            ) + self.weights.get(bias_key)
            if layer_idx == self.L - 1:
                A = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
            else:
                if self.activation == "sig":
                    A = 1 / (1 + np.exp(-z))
                else:
                    A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            self.__cache["A{}".format(layer_idx + 1)] = A

        return A, self.cache

    def cost(self, Y, A):
        """
        Compute the cost function for the logistic function"""
        num_of_sample = Y.shape[1]
        return - np.sum(Y * np.log(A)) / num_of_sample

    def evaluate(self, X, Y):
        """
        Evaluate the model"""
        preds, *_ = self.forward_prop(X)
        cost = self.cost(Y, preds)
        return np.where(preds == np.amax(preds, axis=0), 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Compute the gradient descent for the deep neural network"""
        dZ = self.cache.get("A{}".format(self.L)) - Y
        num_of_sample = Y.shape[1]

        for layer_idx in reversed(range(1, self.L + 1)):
            current_weight_key = "W{}".format(layer_idx)
            current_weight = self.weights.get(current_weight_key)
            current_bias_key = "b{}".format(layer_idx)
            current_bias = self.weights.get(current_bias_key)
            previous_preds_key = "A{}".format(layer_idx - 1)
            previous_preds = self.cache.get(previous_preds_key)

            dW = np.matmul(dZ, previous_preds.T) / num_of_sample
            db = np.sum(dZ, axis=1, keepdims=True) / num_of_sample

            self.__weights[current_weight_key] = current_weight - alpha * dW
            self.__weights[current_bias_key] = current_bias - alpha * db
            if layer_idx > 1:
                if self.activation == "sig":
                    dZ = np.dot(
                        current_weight.T, dZ
                    ) * previous_preds * (1 - previous_preds)
                else:
                    dZ = np.dot(
                        current_weight.T, dZ
                    ) * (1 - previous_preds**2)

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """Train the neural network"""
        _check_iterations(iterations)
        _check_alpha(alpha)
        if graph:
            _check_step(step, iterations)
            cost_list = []
            iter_list = []
        if verbose:
            _check_step(step, iterations)
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, current_cost))
                if graph:
                    cost_list.append(current_cost)
                    iter_list.append(i)
        if graph:
            plt.plot(iter_list, cost_list, "b")
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the instance to a pickle file"""
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Load a model from a file"""
        try:
            with open(filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
