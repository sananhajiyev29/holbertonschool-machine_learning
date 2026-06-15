#!/usr/bin/env python3
"""Module that performs forward propagation for a deep RNN."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances of length l used for
            forward propagation.
        X: data to be used, numpy.ndarray of shape (t, m, i).
        h_0: initial hidden state, numpy.ndarray of shape (l, m, h).

    Returns:
        Tuple of (H, Y) where H contains all hidden states and Y
        contains all outputs.
    """
    t, m, i = X.shape
    layers, _, h = h_0.shape

    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0
    Y = []

    for step in range(t):
        x = X[step]
        for layer in range(layers):
            h_next, y = rnn_cells[layer].forward(H[step, layer], x)
            H[step + 1, layer] = h_next
            x = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
