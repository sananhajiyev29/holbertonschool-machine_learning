#!/usr/bin/env python3
"""Module that creates a variational autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    Args:
        input_dims: integer containing the dimensions of the model input.
        hidden_layers: list containing the number of nodes for each hidden
            layer in the encoder, respectively.
        latent_dims: integer containing the dimensions of the latent space
            representation.

    Returns:
        Tuple of (encoder, decoder, auto).
    """
    X = keras.Input(shape=(input_dims,))
    Y = X
    for n in hidden_layers:
        Y = keras.layers.Dense(n, activation='relu')(Y)
    mean = keras.layers.Dense(latent_dims)(Y)
    log_var = keras.layers.Dense(latent_dims)(Y)

    def sampling(args):
        """Samples z from the latent distribution."""
        mu, lv = args
        eps = keras.backend.random_normal(shape=keras.backend.shape(mu))
        return mu + keras.backend.exp(lv / 2) * eps

    z = keras.layers.Lambda(sampling)([mean, log_var])
    encoder = keras.Model(X, [z, mean, log_var])

    X_dec = keras.Input(shape=(latent_dims,))
    D = X_dec
    for n in reversed(hidden_layers):
        D = keras.layers.Dense(n, activation='relu')(D)
    D = keras.layers.Dense(input_dims, activation='sigmoid')(D)
    decoder = keras.Model(X_dec, D)

    out = decoder(encoder(X)[0])
    auto = keras.Model(X, out)

    def loss(x, x_decoded):
        """Computes the VAE loss."""
        recon = keras.backend.binary_crossentropy(x, x_decoded)
        recon = keras.backend.sum(recon, axis=1)
        kl = -0.5 * keras.backend.sum(
            1 + log_var - keras.backend.square(mean) - keras.backend.exp(
                log_var
            ),
            axis=1
        )
        return recon + kl

    auto.compile(optimizer='adam', loss=loss)

    return encoder, decoder, auto
