#!/usr/bin/env python3
"""Module that performs back propagation over a convolutional layer."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new).
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev).
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new).
        b: numpy.ndarray of shape (1, 1, 1, c_new).
        padding: 'same' or 'valid'.
        stride: tuple of (sh, sw).

    Returns:
        dA_prev, dW, db respectively.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        ph_total = (h_new - 1) * sh + kh - h_prev
        pw_total = (w_new - 1) * sw + kw - w_prev
        ph_top = (ph_total + 1) // 2
        ph_bot = ph_total - ph_top
        pw_left = (pw_total + 1) // 2
        pw_right = pw_total - pw_left
    else:
        ph_top = ph_bot = pw_left = pw_right = 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph_top, ph_bot), (pw_left, pw_right), (0, 0)),
        mode='constant'
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            w_start = j * sw
            region = A_prev_pad[:, h_start:h_start + kh,
                                 w_start:w_start + kw, :]
            dz = dZ[:, i, j, :]
            dA_prev_pad[:, h_start:h_start + kh,
                        w_start:w_start + kw, :] += np.tensordot(
                dz, W, axes=[[1], [3]]
            )
            dW += np.tensordot(region, dz, axes=[[0], [0]])

    if padding == 'same':
        h_end = A_prev_pad.shape[1] - ph_bot if ph_bot > 0 else None
        w_end = A_prev_pad.shape[2] - pw_right if pw_right > 0 else None
        dA_prev = dA_prev_pad[:, ph_top:h_end, pw_left:w_end, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
