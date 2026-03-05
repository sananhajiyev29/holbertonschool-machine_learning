#!/usr/bin/env python3
"""Deep Neural Network with train method"""

import numpy as np
from 21-deep_neural_network import DeepNeuralNetwork


class DeepNeuralNetwork(DeepNeuralNetwork):
    """Deep Neural Network performing binary classification with training"""

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
