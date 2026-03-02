#!/usr/bin/env python3
"""16-deep_neural_network.py"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initialize a deep neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            layer_size = layers[l]
            if l == 0:
                prev_size = nx
            else:
                prev_size = layers[l - 1]

            self.weights["W{}".format(l + 1)] = (np.random.randn(layer_size,
                prev_size) * np.sqrt(2 / prev_size))
            self.weights["b{}".format(l + 1)] = np.zeros((layer_size, 1))
