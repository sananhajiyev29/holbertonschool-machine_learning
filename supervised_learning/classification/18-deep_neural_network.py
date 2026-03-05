#!/usr/bin/env python3
"""Deep Neural Network class for binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for neurons in layers:
            if not isinstance(neurons, int) or neurons < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            layer_num = i + 1

            if i == 0:
                self.__weights['W{}'.format(layer_num)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W{}'.format(layer_num)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            self.__weights['b{}'.format(layer_num)] = np.zeros((layers[i], 1))

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
        """Calculates forward propagation of the neural network"""
        self.__cache['A0'] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(layer)]
            b = self.__weights['b{}'.format(layer)]
            A_prev = self.__cache['A{}'.format(layer - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache['A{}'.format(layer)] = A

        return A, self.__cache
