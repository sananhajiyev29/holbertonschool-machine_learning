#!/usr/bin/env python3
"""Deep Neural Network forward propagation"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx

        for i in range(self.__L):

            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            nodes = layers[i]

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )

            self.__weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

            prev = nodes

    @property
    def L(self):
        """getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation
        """

        self.__cache["A0"] = X

        for l in range(1, self.__L + 1):

            W = self.__weights["W{}".format(l)]
            b = self.__weights["b{}".format(l)]
            A_prev = self.__cache["A{}".format(l - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(l)] = A

        return A, self.__cache
