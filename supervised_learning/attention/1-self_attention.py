#!/usr/bin/env python3
"""Module that calculates attention for machine translation."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates the attention for machine translation."""

    def __init__(self, units):
        """Initializes the SelfAttention layer.

        Args:
            units: integer representing the number of hidden units in the
                alignment model.
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculates the context vector and attention weights.

        Args:
            s_prev: tensor of shape (batch, units) containing the previous
                decoder hidden state.
            hidden_states: tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder.

        Returns:
            Tuple of (context, weights) where context is a tensor of shape
            (batch, units) and weights is a tensor of shape
            (batch, input_seq_len, 1).
        """
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)

        score = self.V(tf.nn.tanh(
            self.W(s_prev_expanded) + self.U(hidden_states)
        ))

        weights = tf.nn.softmax(score, axis=1)

        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
