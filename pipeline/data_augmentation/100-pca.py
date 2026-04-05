#!/usr/bin/env python3
"""Module that performs PCA color augmentation as in the AlexNet paper."""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: 3D tf.Tensor containing the image to change.
        alphas: tuple of length 3 with the amount each channel should change.

    Returns:
        The augmented image.
    """
    image = tf.cast(image, tf.float64)
    img = image.numpy()

    img_flat = img.reshape(-1, 3)
    img_norm = img_flat / 255.0

    mean = np.mean(img_norm, axis=0)
    img_centered = img_norm - mean

    cov = np.cov(img_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    delta = np.dot(
        eigenvectors,
        (alphas * eigenvalues)
    )

    img_augmented = img_norm + delta
    img_augmented = np.clip(img_augmented, 0.0, 1.0)
    img_augmented = (img_augmented * 255.0).reshape(img.shape)

    return tf.cast(tf.convert_to_tensor(img_augmented), tf.uint8)
