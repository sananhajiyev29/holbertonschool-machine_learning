#!/usr/bin/env python3
"""Module for gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases using gradient descent with L2 reg.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with labels.
        weights: dictionary of weights and biases of the neural network.
        cache: dictionary of outputs of each layer of the neural network.
        alpha: the learning rate.
        lambtha: the L2 regularization parameter.
        L: the number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    weights_copy = weights.copy()

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]

        dW = np.matmul(dZ, A_prev.T) / m + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] -= alpha * db
