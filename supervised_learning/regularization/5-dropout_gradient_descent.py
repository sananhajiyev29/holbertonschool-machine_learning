#!/usr/bin/env python3
"""Module that updates weights using gradient descent with Dropout."""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights of a neural network with Dropout using gradient descent.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels.
        weights: dictionary of the weights and biases of the neural network.
        cache: dictionary of outputs and dropout masks of each layer.
        alpha: the learning rate.
        keep_prob: the probability that a node will be kept.
        L: the number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    weights_copy = weights.copy()

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA = np.matmul(W.T, dZ)
            dA = dA * cache['D' + str(i - 1)] / keep_prob
            dZ = dA * (1 - cache['A' + str(i - 1)] ** 2)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
