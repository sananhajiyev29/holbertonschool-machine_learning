#!/usr/bin/env python3
"""Module that defines a single neuron for binary classification"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """Calculates forward propagation"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates logistic regression cost"""
        m = Y.shape[1]
        return - (1 / m) * np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Performs one pass of gradient descent"""
        m = Y.shape[1]

        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        Returns the evaluation of the training data
        """

        # Validate iterations
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # Validate alpha
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
