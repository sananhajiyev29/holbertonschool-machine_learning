#!/usr/bin/env python3
"""Module that performs PCA color augmentation as in the AlexNet paper."""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: 3D tf.Tensor containing the image to change.
        alphas: tuple of length 3 with the amount each channel should change.

    Returns:
        The augmented image.
    """
    image = tf.cast(image, tf.float64)
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]

    img_flat = tf.reshape(image, [-1, 3])
    img_norm = img_flat / 255.0

    mean = tf.reduce_mean(img_norm, axis=0, keepdims=True)
    img_centered = img_norm - mean

    n = tf.cast(tf.shape(img_centered)[0] - 1, tf.float64)
    cov = tf.matmul(tf.transpose(img_centered), img_centered) / n

    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    idx = tf.argsort(eigenvalues, direction='DESCENDING')
    eigenvalues = tf.gather(eigenvalues, idx)
    eigenvectors = tf.gather(eigenvectors, idx, axis=1)

    alphas = tf.cast(alphas, tf.float64)
    delta = tf.linalg.matvec(eigenvectors, alphas * eigenvalues)

    img_augmented = img_norm + delta
    img_augmented = tf.clip_by_value(img_augmented, 0.0, 1.0)
    img_augmented = img_augmented * 255.0
    img_augmented = tf.reshape(img_augmented, [h, w, 3])

    return tf.cast(img_augmented, tf.uint8)
