#!/usr/bin/env python3
"""
Module 21-deep_neural_network
Defines DeepNeuralNetwork class with gradient descent.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initialize network.
        nx: number of input features
        layers: list of nodes per layer
        """
        if type(nx) is not int or nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0 or not all(
            type(x) == int and x > 0 for x in layers
        ):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.__weights = {}

        for l in range(self.L):
            if l == 0:
                self.__weights["W1"] = (
                    np.random.randn(layers[l], nx) * np.sqrt(2 / nx)
                )
            else:
                self.__weights["W" + str(l + 1)] = (
                    np.random.randn(layers[l], layers[l - 1])
                    * np.sqrt(2 / layers[l - 1])
                )
            self.__weights["b" + str(l + 1)] = np.zeros((layers[l], 1))

    @property
    def weights(self):
        """Get network weights."""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation for the network."""
        self.cache["A0"] = X
        for l in range(1, self.L + 1):
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]
            A_prev = self.cache["A" + str(l - 1)]
            Z = np.dot(W, A_prev) + b
            self.cache["A" + str(l)] = 1 / (1 + np.exp(-Z))
        return self.cache["A" + str(self.L)], self.cache

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent.
        Y: correct labels, shape (1, m)
        cache: forward propagation cache
        alpha: learning rate
        """
        m = Y.shape[1]
        L = self.L
        dZ = cache["A" + str(L)] - Y

        for l in reversed(range(1, L + 1)):
            A_prev = cache["A" + str(l - 1)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["W" + str(l)] -= alpha * dW
            self.__weights["b" + str(l)] -= alpha * db
            if l > 1:
                A_prev = cache["A" + str(l - 1)]
                dZ = np.dot(self.__weights["W" + str(l)].T, dZ) * (A_prev * (1 - A_prev))
