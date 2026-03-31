#!/usr/bin/env python3
"""Module that performs back propagation over a pooling layer."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new).
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c).
        kernel_shape: tuple of (kh, kw) for the pooling kernel size.
        stride: tuple of (sh, sw) for height and width strides.
        mode: 'max' or 'avg' pooling.

    Returns:
        dA_prev: partial derivatives with respect to the previous layer.
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            w_start = j * sw
            if mode == 'max':
                region = A_prev[:, h_start:h_start + kh,
                                 w_start:w_start + kw, :]
                mask = (region == np.max(
                    region, axis=(1, 2), keepdims=True
                ))
                dA_prev[:, h_start:h_start + kh,
                        w_start:w_start + kw, :] += (
                    mask * dA[:, i, j, :][:, np.newaxis, np.newaxis, :]
                )
            else:
                avg = dA[:, i, j, :] / (kh * kw)
                dA_prev[:, h_start:h_start + kh,
                        w_start:w_start + kw, :] += np.ones(
                    (m, kh, kw, c)
                ) * avg[:, np.newaxis, np.newaxis, :]

    return dA_prev
