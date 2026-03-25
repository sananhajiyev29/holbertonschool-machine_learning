#!/usr/bin/env python3
"""Module that performs a valid convolution on grayscale images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray of shape (m, h, w) with grayscale images.
        kernel: numpy.ndarray of shape (kh, kw) containing the kernel.

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    oh = h - kh + 1
    ow = w - kw + 1
    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                images[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )
    return output
