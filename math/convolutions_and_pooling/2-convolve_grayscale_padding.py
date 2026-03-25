#!/usr/bin/env python3
"""Module that performs a convolution on grayscale images with padding."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding.

    Args:
        images: numpy.ndarray of shape (m, h, w) with grayscale images.
        kernel: numpy.ndarray of shape (kh, kw) containing the kernel.
        padding: tuple of (ph, pw) for height and width padding.

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )
    return output
