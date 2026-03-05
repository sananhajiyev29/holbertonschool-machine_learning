#!/usr/bin/env python3
"""Deep Neural Network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class"""

    def __init__(self, nx, layers):
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
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation using sigmoid activation"""
        self.__cache["A0"] = X
        for l in range(1, self.__L + 1):
            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]
            A_prev = self.__cache[f"A{l-1}"]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f"A{l}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Compute cost using logistic regression"""
        m = Y.shape[1]
        return -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform one pass of gradient descent"""
        m = Y.shape[1]
        L = self.__L
        dZ = cache[f"A{L}"] - Y

        for l in reversed(range(1, L + 1)):
            A_prev = cache[f"A{l-1}"]
            W = self.__weights[f"W{l}"]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f"W{l}"] -= alpha * dW
            self.__weights[f"b{l}"] -= alpha * db

            if l > 1:
                A_prev_layer = cache[f"A{l-1}"]
                dZ = np.matmul(W.T, dZ) * (A_prev_layer * (1 - A_prev_layer))
