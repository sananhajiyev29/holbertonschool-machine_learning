#!/usr/bin/env python3
"""Module that rotates an image by 90 degrees counter-clockwise."""
import tensorflow as tf


def rotate_image(image):
    """Rotates an image by 90 degrees counter-clockwise.

    Args:
        image: 3D tf.Tensor containing the image to rotate.

    Returns:
        The rotated image.
    """
    return tf.image.rot90(image)
