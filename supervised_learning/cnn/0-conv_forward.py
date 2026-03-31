#!/usr/bin/env python3
"""Module that performs forward propagation over a convolutional layer."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev).
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) with kernels.
        b: numpy.ndarray of shape (1, 1, 1, c_new) with biases.
        activation: activation function applied to the convolution.
        padding: 'same' or 'valid'.
        stride: tuple of (sh, sw) for height and width strides.

    Returns:
        The output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((h_prev - 1) * sh - h_prev + kh, 0) // 2
        pw = max((w_prev - 1) * sw - w_prev + kw, 0) // 2
    else:
        ph, pw = 0, 0

    oh = (h_prev + 2 * ph - kh) // sh + 1
    ow = (w_prev + 2 * pw - kw) // sw + 1

    padded = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
    )

    Z = np.zeros((m, oh, ow, c_new))
    for i in range(oh):
        for j in range(ow):
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(
                    padded[:, i * sh:i * sh + kh,
                           j * sw:j * sw + kw, :] *
                    W[:, :, :, k],
                    axis=(1, 2, 3)
                ) + b[0, 0, 0, k]

    return activation(Z)
