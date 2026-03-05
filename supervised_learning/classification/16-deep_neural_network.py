#!/usr/bin/env python3
"""Deep Neural Network class"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev = nx

        for i in range(self.L):

            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )

            nodes = layers[i]

            self.weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )

            self.weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

            prev = nodes
