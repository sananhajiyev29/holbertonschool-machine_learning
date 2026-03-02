#!/usr/bin/env python3
"""DeepNeuralNetwork class with private attributes and getters"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initialize the deep neural network

        Args:
            nx (int): number of input features
            layers (list): list of nodes in each layer
        """
        # Input validation
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Single loop to initialize weights and biases using He et al. method
        prev_size = nx
        for l, nodes in enumerate(layers, start=1):
            self.__weights[f"W{l}"] = np.random.randn(nodes, prev_size) * np.sqrt(2 / prev_size)
            self.__weights[f"b{l}"] = np.zeros((nodes, 1))
            prev_size = nodes

    # Getters
    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Dictionary to store intermediate values"""
        return self.__cache

    @property
    def weights(self):
        """Dictionary to store weights and biases"""
        return self.__weights
