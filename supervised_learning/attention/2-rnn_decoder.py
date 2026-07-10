#!/usr/bin/env python3
"""Module that defines an RNN decoder for machine translation."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes for machine translation using an RNN."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the RNNDecoder.

        Args:
            vocab: integer representing the size of the output vocabulary.
            embedding: integer representing the dimensionality of the
                embedding vector.
            units: integer representing the number of hidden units in the
                RNN cell.
            batch: integer representing the batch size.
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Performs the forward pass of the decoder.

        Args:
            x: tensor of shape (batch, 1) containing the previous word in
                the target sequence as an index of the target vocabulary.
            s_prev: tensor of shape (batch, units) containing the previous
                decoder hidden state.
            hidden_states: tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder.

        Returns:
            Tuple of (y, s) where y is a tensor of shape (batch, vocab)
            and s is a tensor of shape (batch, units).
        """
        units = s_prev.shape[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)

        x = self.embedding(x)

        context_expanded = tf.expand_dims(context, axis=1)
        x = tf.concat([context_expanded, x], axis=-1)

        outputs, s = self.gru(x)

        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        y = self.F(outputs)

        return y, s
