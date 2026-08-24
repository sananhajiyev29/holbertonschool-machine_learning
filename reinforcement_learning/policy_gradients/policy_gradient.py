#!/usr/bin/env python3
"""Module that computes the Monte-Carlo policy gradient."""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with a weight of a matrix.

    Args:
        matrix: the state matrix.
        weight: the weight matrix.

    Returns:
        The policy as a numpy.ndarray of action probabilities.
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z))

    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """Computes the Monte-Carlo policy gradient.

    Args:
        state: matrix representing the current observation of the
            environment.
        weight: matrix of random weight.

    Returns:
        Tuple of (action, gradient).
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)

    action = np.random.choice(probs.shape[1], p=probs[0])

    softmax = probs.reshape(-1, 1)
    d_softmax = np.diagflat(softmax) - np.dot(softmax, softmax.T)
    d_log = d_softmax[action] / softmax[action]

    gradient = state.T.dot(d_log.reshape(1, -1))

    return action, gradient
