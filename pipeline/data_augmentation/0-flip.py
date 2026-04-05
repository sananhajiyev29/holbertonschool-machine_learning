#!/usr/bin/env python3
"""Module that flips an image horizontally."""
import tensorflow as tf


def flip_image(image):
    """Flips an image horizontally.

    Args:
        image: 3D tf.Tensor containing the image to flip.

    Returns:
        The flipped image.
    """
    return tf.image.flip_left_right(image)
