#!/usr/bin/env python3
"""Module that performs pooling on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images.

    Args:
        images: numpy.ndarray of shape (m, h, w, c) with images.
        kernel_shape: tuple of (kh, kw) for the pooling kernel shape.
        stride: tuple of (sh, sw) for height and width strides.
        mode: 'max' for max pooling or 'avg' for average pooling.

    Returns:
        numpy.ndarray containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    output = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            patch = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(patch, axis=(1, 2))

    return output
