#!/usr/bin/env python3
"""Module that creates a variational autoencoder."""
import tensorflow.keras as keras
import tensorflow.keras.backend as K


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
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Samples from the latent distribution."""
        mu, lv = args
        epsilon = K.random_normal(shape=K.shape(mu))
        return mu + K.exp(lv / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mean, log_var])
    encoder = keras.Model(encoder_input, [z, mean, log_var])

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded[0])
    auto = keras.Model(auto_input, decoded)

    def vae_loss(y_true, y_pred):
        """Computes the VAE loss."""
        reconstruction = keras.losses.binary_crossentropy(y_true, y_pred)
        reconstruction = reconstruction * input_dims
        kl = -0.5 * K.sum(
            1 + log_var - K.square(mean) - K.exp(log_var), axis=-1
        )
        return reconstruction + kl

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
