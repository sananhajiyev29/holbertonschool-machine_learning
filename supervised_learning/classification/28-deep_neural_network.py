#!/usr/bin/env python3
"""Module that defines a deep neural network for multiclass classification."""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification."""

    def __init__(self, nx, layers, activation='sig'):
        """Initialize DeepNeuralNetwork.

        Args:
            nx: number of input features.
            layers: list representing the number of nodes in each layer.
            activation: type of activation function in hidden layers.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights['W' + str(i + 1)] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                )
            else:
                self.__weights['W' + str(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2 / layers[i - 1])
                )
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for L."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights."""
        return self.__weights

    @property
    def activation(self):
        """Getter for activation."""
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network.

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data.

        Returns:
            Output of the neural network and the cache.
        """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            if i == self.__L:
                t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache['A' + str(i)] = t / np.sum(
                    t, axis=0, keepdims=True
                )
            elif self.__activation == 'sig':
                self.__cache['A' + str(i)] = 1 / (1 + np.exp(-Z))
            else:
                self.__cache['A' + str(i)] = np.tanh(Z)
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost using softmax cross-entropy.

        Args:
            Y: one-hot numpy.ndarray with shape (classes, m) correct labels.
            A: numpy.ndarray with shape (classes, m) with activated output.

        Returns:
            The cost.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions.

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data.
            Y: one-hot numpy.ndarray with shape (classes, m) correct labels.

        Returns:
            Tuple of the predicted labels and the cost.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A == np.max(A, axis=0, keepdims=True), 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network.

        Args:
            Y: one-hot numpy.ndarray with shape (classes, m) correct labels.
            cache: dictionary containing all intermediary values of network.
            alpha: the learning rate.
        """
        m = Y.shape[1]
        dZ = cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = self.__weights['W' + str(i)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                if self.__activation == 'sig':
                    dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))
                else:
                    dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)
            else:
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))

            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network.

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data.
            Y: one-hot numpy.ndarray with shape (classes, m) correct labels.
            iterations: number of iterations to train over.
            alpha: the learning rate.
            verbose: boolean whether to print training info.
            graph: boolean whether to graph training info.
            step: interval for printing/graphing.

        Returns:
            Evaluation of training data after iterations of training.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)

        if verbose:
            print("Cost after 0 iterations: {}".format(cost))
        if graph:
            costs.append(cost)
            iters.append(0)

        for i in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if verbose and i % step == 0:
                A, cache = self.forward_prop(X)
                cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i, cost))
            if graph and i % step == 0:
                A, cache = self.forward_prop(X)
                cost = self.cost(Y, A)
                costs.append(cost)
                iters.append(i)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format.

        Args:
            filename: file to which the object should be saved.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object.

        Args:
            filename: file from which the object should be loaded.

        Returns:
            The loaded object, or None if filename doesn't exist.
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
