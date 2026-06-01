#!/usr/bin/env python3
"""Module that creates a convolutional autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder.

    Args:
        input_dims: tuple of integers containing the dimensions of the
            model input.
        filters: list containing the number of filters for each
            convolutional layer in the encoder, respectively.
        latent_dims: tuple of integers containing the dimensions of the
            latent space representation.

    Returns:
        Tuple of (encoder, decoder, auto).
    """
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(encoder_input, x)

    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    for f in reversed(filters[1:]):
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(
        filters[0], (3, 3), padding='valid', activation='relu'
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same', activation='sigmoid'
    )(x)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
