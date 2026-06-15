#!/usr/bin/env python3
"""Module that performs forward propagation for a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: instance of RNNCell used for forward propagation.
        X: data to be used, numpy.ndarray of shape (t, m, i).
        h_0: initial hidden state, numpy.ndarray of shape (m, h).

    Returns:
        Tuple of (H, Y) where H contains all hidden states and Y
        contains all outputs.
    """
    t, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    Y = []

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
