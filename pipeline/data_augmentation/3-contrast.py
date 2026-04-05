#!/usr/bin/env python3
"""Module that randomly adjusts the contrast of an image."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjusts the contrast of an image.

    Args:
        image: 3D tf.Tensor representing the input image.
        lower: float representing the lower bound of the contrast factor.
        upper: float representing the upper bound of the contrast factor.

    Returns:
        The contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower, upper)
