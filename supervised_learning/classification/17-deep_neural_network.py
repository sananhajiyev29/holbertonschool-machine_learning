#!/usr/bin/env python3
"""Deep Neural Network with private attributes"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""

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
                raise TypeError(
                    "layers must be a list of positive integers"
                )

            nodes = layers[i]

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )

            self.__weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

            prev = nodes

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights
