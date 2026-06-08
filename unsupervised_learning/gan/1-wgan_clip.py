#!/usr/bin/env python3
"""Module that defines a Wasserstein GAN with weight clipping."""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class WGAN_clip(keras.Model):
    """A Wasserstein GAN that clips the discriminator weights."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initializes the WGAN_clip model.

        Args:
            generator: the generator network.
            discriminator: the discriminator network.
            latent_generator: function generating latent vectors.
            real_examples: tensor of real examples.
            batch_size: the batch size.
            disc_iter: number of discriminator iterations per step.
            learning_rate: the learning rate.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        self.discriminator.loss = lambda x, y: (
            tf.math.reduce_mean(y) - tf.math.reduce_mean(x)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Generates a fake sample using the generator.

        Args:
            size: number of samples to generate.
            training: whether the generator is in training mode.

        Returns:
            A batch of generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Gets a random real sample.

        Args:
            size: number of real samples to return.

        Returns:
            A batch of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """Performs one training step of the WGAN.

        Args:
            useless_argument: required by the Keras API but unused.

        Returns:
            A dictionary with the discriminator and generator losses.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)
                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)
                discr_loss = self.discriminator.loss(real_output, fake_output)
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(fake_output)
        grads = tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
