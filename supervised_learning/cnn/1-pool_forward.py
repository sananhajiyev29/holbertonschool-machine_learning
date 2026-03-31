#!/usr/bin/env python3
"""Module that performs forward propagation over a pooling layer."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev).
        kernel_shape: tuple of (kh, kw) for the pooling kernel size.
        stride: tuple of (sh, sw) for height and width strides.
        mode: 'max' or 'avg' pooling.

    Returns:
        The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h_prev - kh) // sh + 1
    ow = (w_prev - kw) // sw + 1

    A = np.zeros((m, oh, ow, c_prev))

    for i in range(oh):
        for j in range(ow):
            region = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                A[:, i, j, :] = np.max(region, axis=(1, 2))
            else:
                A[:, i, j, :] = np.mean(region, axis=(1, 2))

    return A
