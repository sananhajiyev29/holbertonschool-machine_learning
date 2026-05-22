#!/usr/bin/env python3
"""Module that performs tasks for neural style transfer."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializes the NST class.

        Args:
            style_image: numpy.ndarray with the style reference image.
            content_image: numpy.ndarray with the content reference image.
            alpha: the weight for content cost.
            beta: the weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Rescales an image to pixel values [0, 1] with max side 512.

        Args:
            image: numpy.ndarray of shape (h, w, 3) containing the image.

        Returns:
            The scaled image as a tf.Tensor of shape (1, h_new, w_new, 3).
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(
            image, (h_new, w_new), method=tf.image.ResizeMethod.BICUBIC
        )
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def load_model(self):
        """Creates the model used to calculate cost.

        The model uses VGG19 as a base. The model's input is the same as
        VGG19's input. The model's output is a list of the outputs of the
        style layers followed by the content layer.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )
        vgg.trainable = False

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg.save("vgg_base.h5")
        vgg = tf.keras.models.load_model(
            "vgg_base.h5", custom_objects=custom_objects
        )

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers
        ]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer.

        Args:
            input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c).

        Returns:
            A tf.Tensor of shape (1, c, c) containing the gram matrix.
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / num_locations
