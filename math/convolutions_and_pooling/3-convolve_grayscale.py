#!/usr/bin/env python3
"""Module that performs a convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray of shape (m, h, w) with grayscale images.
        kernel: numpy.ndarray of shape (kh, kw) containing the kernel.
        padding: 'same', 'valid', or tuple (ph, pw).
        stride: tuple of (sh, sw) for height and width strides.

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel,
                axis=(1, 2)
            )
    return output
