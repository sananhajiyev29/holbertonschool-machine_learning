#!/usr/bin/env python3
"""Module that performs tasks for neural style transfer."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1,
                 var=10):
        """Initializes the NST class.

        Args:
            style_image: numpy.ndarray with the style reference image.
            content_image: numpy.ndarray with the content reference image.
            alpha: the weight for content cost.
            beta: the weight for style cost.
            var: the weight for variational cost.
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
        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("var must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

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

    def generate_features(self):
        """Extracts the features used to calculate neural style cost.

        Sets the public instance attributes:
            gram_style_features: list of gram matrices from style layer
                outputs of the style image.
            content_feature: content layer output of the content image.
        """
        style_input = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content_input = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_input)[:-1]
        content_output = self.model(content_input)[-1]

        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs
        ]
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer.

        Args:
            style_output: tf.Tensor of shape (1, h, w, c) with the layer
                style output of the generated image.
            gram_target: tf.Tensor of shape (1, c, c) with the gram matrix
                of the target style output for that layer.

        Returns:
            The layer's style cost.
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c
                )
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculates the style cost for the generated image.

        Args:
            style_outputs: list of tf.Tensor style outputs for the
                generated image.

        Returns:
            The style cost.
        """
        length = len(self.style_layers)
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != length):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length
                )
            )

        weight = 1 / length
        cost = 0
        for i in range(length):
            cost += weight * self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i]
            )

        return cost

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image.

        Args:
            content_output: tf.Tensor containing the content output for
                the generated image.

        Returns:
            The content cost.
        """
        s = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != s):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s)
            )

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    @staticmethod
    def variational_cost(generated_image):
        """Calculates the variational cost for the generated image.

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3) containing
                the generated image.

        Returns:
            The variational cost.
        """
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                len(generated_image.shape) != 4):
            raise TypeError("generated_image must be a tensor of rank 4")

        return tf.reduce_sum(tf.image.total_variation(generated_image))

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image.

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3) containing
                the generated image.

        Returns:
            Tuple of (J, J_content, J_style, J_var).
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s)
            )

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )
        outputs = self.model(preprocessed)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J_var = self.variational_cost(generated_image)
        J = (self.alpha * J_content + self.beta * J_style +
             self.var * J_var)

        return J, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Calculates gradients for the generated image.

        Args:
            generated_image: tf.Tensor or tf.Variable of shape
                (1, nh, nw, 3) containing the generated image.

        Returns:
            Tuple of (gradients, J_total, J_content, J_style, J_var).
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s)
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image
            )

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the neural style transferred image.

        Args:
            iterations: number of iterations to perform gradient descent.
            step: optional interval for printing training costs.
            lr: learning rate for Adam optimization.
            beta1: beta1 parameter for Adam optimization.
            beta2: beta2 parameter for Adam optimization.

        Returns:
            Tuple of (generated_image, cost), where generated_image is the
            best generated image and cost is the best cost.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step is not None and (step <= 0 or step >= iterations):
            raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2
        )
        best_cost = np.inf
        best_image = None

        for i in range(iterations + 1):
            gradients, J_total, J_content, J_style, J_var = (
                self.compute_grads(generated_image)
            )
            cost = J_total.numpy()

            if cost < best_cost:
                best_cost = cost
                best_image = generated_image.numpy()[0]

            if step is not None and (i % step == 0 or i == iterations):
                print(
                    "Cost at iteration {}: {}, content {}, style {}, "
                    "var {}".format(
                        i, cost, J_content.numpy(), J_style.numpy(),
                        J_var.numpy()
                    )
                )

            if i < iterations:
                optimizer.apply_gradients([(gradients, generated_image)])
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0)
                )

        return best_image, best_cost
