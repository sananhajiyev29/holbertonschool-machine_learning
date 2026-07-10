#!/usr/bin/env python3
"""Module that performs multi head attention."""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention."""

    def __init__(self, dm, h):
        """Initializes the MultiHeadAttention layer.

        Args:
            dm: integer representing the dimensionality of the model.
            h: integer representing the number of heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch):
        """Splits the last dimension into (h, depth) and transposes.

        Args:
            x: tensor to split.
            batch: the batch size.

        Returns:
            The reshaped and transposed tensor of shape
            (batch, h, seq_len, depth).
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Performs multi head attention.

        Args:
            Q: tensor of shape (batch, seq_len_q, dk) for the query input.
            K: tensor of shape (batch, seq_len_v, dk) for the key input.
            V: tensor of shape (batch, seq_len_v, dv) for the value input.
            mask: always None.

        Returns:
            Tuple of (output, weights) where output has its last two
            dimensions as (..., seq_len_q, dm) and weights has its last
            three dimensions as (..., h, seq_len_q, seq_len_v).
        """
        batch = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch)
        K = self.split_heads(K, batch)
        V = self.split_heads(V, batch)

        attention, weights = sdp_attention(Q, K, V, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(attention, (batch, -1, self.dm))

        output = self.linear(concat)

        return output, weights
