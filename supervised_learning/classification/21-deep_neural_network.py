#!/usr/bin/env python3
"""Deep Neural Network with gradient descent"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initialize the network"""
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
            self.__weights[f"W{i+1}"] = np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            self.__weights[f"b{i+1}"] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Cache of activated outputs"""
        return self.__cache

    @property
    def weights(self):
        """Weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation with sigmoid activation"""
        self.__cache["A0"] = X
        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            A_prev = self.__cache[f"A{layer-1}"]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f"A{layer}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Compute logistic regression cost"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions and cost"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform one pass of gradient descent"""
        m = Y.shape[1]
        L = self.__L
        dZ = cache[f"A{L}"] - Y

        for layer in reversed(range(1, L + 1)):
            A_prev = cache[f"A{layer-1}"]
            W = self.__weights[f"W{layer}"]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f"W{layer}"] -= alpha * dW
            self.__weights[f"b{layer}"] -= alpha * db

            if layer > 1:
                A_prev_layer = cache[f"A{layer-1}"]
                dZ = np.matmul(W.T, dZ) * (A_prev_layer * (1 - A_prev_layer))
