#!/usr/bin/env python3
"""Module that defines an RNN encoder for machine translation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodes for machine translation using an RNN."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the RNNEncoder.

        Args:
            vocab: integer representing the size of the input vocabulary.
            embedding: integer representing the dimensionality of the
                embedding vector.
            units: integer representing the number of hidden units in the
                RNN cell.
            batch: integer representing the batch size.
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to zeros.

        Returns:
            A tensor of shape (batch, units) containing the initialized
            hidden states.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Performs the forward pass of the encoder.

        Args:
            x: tensor of shape (batch, input_seq_len) containing the input
                to the encoder as word indices within the vocabulary.
            initial: tensor of shape (batch, units) containing the initial
                hidden state.

        Returns:
            Tuple of (outputs, hidden) where outputs is a tensor of shape
            (batch, input_seq_len, units) and hidden is a tensor of shape
            (batch, units).
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
