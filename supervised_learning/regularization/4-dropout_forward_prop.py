#!/usr/bin/env python3
"""Module that conducts forward propagation using Dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout.

    Args:
        X: numpy.ndarray of shape (nx, m) containing the input data.
        weights: dictionary of the weights and biases of the neural network.
        L: the number of layers in the network.
        keep_prob: the probability that a node will be kept.

    Returns:
        Dictionary containing the outputs of each layer and dropout masks.
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            cache['D' + str(i)] = D
            cache['A' + str(i)] = (A * D) / keep_prob

    return cache
