#!/usr/bin/env python3
"""Module that performs a same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray of shape (m, h, w) with grayscale images.
        kernel: numpy.ndarray of shape (kh, kw) containing the kernel.

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )
    return output
