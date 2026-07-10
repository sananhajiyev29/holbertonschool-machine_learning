#!/usr/bin/env python3
"""Module that creates a transformer network."""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Creates a transformer network."""

    def __init__(self, N, dm, h, hidden,
                 input_vocab, target_vocab,
                 max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """Initialize the transformer."""
        super().__init__()

        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )

        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask,
             decoder_mask):
        """Forward propagation for the transformer.

        Args:
            inputs: Tensor containing the encoder inputs.
            target: Tensor containing the decoder inputs.
            training: Boolean indicating training mode.
            encoder_mask: Padding mask for the encoder.
            look_ahead_mask: Look-ahead mask for the decoder.
            decoder_mask: Padding mask for the decoder.

        Returns:
            Tensor of shape (batch, target_seq_len, target_vocab).
        """
        encoder_output = self.encoder(
            inputs,
            training,
            encoder_mask
        )

        decoder_output = self.decoder(
            target,
            encoder_output,
            training,
            look_ahead_mask,
            decoder_mask
        )

        return self.linear(decoder_output)
