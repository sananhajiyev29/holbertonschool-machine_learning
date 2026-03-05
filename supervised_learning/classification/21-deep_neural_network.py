#!/usr/bin/env python3
"""Deep Neural Network with forward propagation, cost, evaluation, and gradient descent"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for i, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.__weights["b{}".format(i + 1)] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation"""
        self.__cache["A0"] = X
        for layer in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            A_prev = self.__cache["A{}".format(layer - 1)]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache["A{}".format(layer)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost using logistic regression"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions and calculates cost"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs one pass of gradient descent on the network"""
        m = Y.shape[1]
        L = self.__L
        weights_copy = self.__weights.copy()

        # Initialize derivative of activation for last layer
        dZ = cache["A{}".format(L)] - Y

        for layer in reversed(range(1, L + 1)):
            A_prev = cache["A{}".format(layer - 1)]
            W = weights_copy["W{}".format(layer)]
            # Gradients
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            # Update weights and biases
            self.__weights["W{}".format(layer)] -= alpha * dW
            self.__weights["b{}".format(layer)] -= alpha * db
            if layer > 1:
                A_prev_layer = cache["A{}".format(layer - 1)]
                dZ = np.matmul(weights_copy["W{}".format(layer)].T, dZ) * (
                    A_prev_layer * (1 - A_prev_layer)
                )
