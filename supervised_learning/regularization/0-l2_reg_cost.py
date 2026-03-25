#!/usr/bin/env python3
"""Module that calculates the cost of a neural network with L2 regularization."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: the cost of the network without L2 regularization.
        lambtha: the regularization parameter.
        weights: dictionary of the weights and biases of the neural network.
        L: the number of layers in the neural network.
        m: the number of data points used.

    Returns:
        The cost of the network accounting for L2 regularization.
    """
    frobenius = 0
    for i in range(1, L + 1):
        frobenius += np.sum(np.square(weights['W' + str(i)]))
    return cost + (lambtha / (2 * m)) * frobenius
