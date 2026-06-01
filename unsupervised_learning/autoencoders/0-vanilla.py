#!/usr/bin/env python3
"""Module that creates a vanilla autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a vanilla autoencoder.

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
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(encoder_input, latent)

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
