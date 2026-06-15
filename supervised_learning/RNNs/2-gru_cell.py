#!/usr/bin/env python3
"""Module that defines a gated recurrent unit (GRU) cell."""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit."""

    def __init__(self, i, h, o):
        """Initializes the GRUCell.

        Args:
            i: dimensionality of the data.
            h: dimensionality of the hidden state.
            o: dimensionality of the outputs.
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with the previous
                hidden state.
            x_t: numpy.ndarray of shape (m, i) with the data input.

        Returns:
            Tuple of (h_next, y) where h_next is the next hidden state
            and y is the output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_candidate = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_candidate

        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, y
