#!/usr/bin/env python3
"""Module that creates a decoder block for a transformer."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Creates a decoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the DecoderBlock.

        Args:
            dm: the dimensionality of the model.
            h: the number of heads.
            hidden: the number of hidden units in the fully connected
                layer.
            drop_rate: the dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Performs the forward pass of the decoder block.

        Args:
            x: tensor of shape (batch, target_seq_len, dm) containing the
                input to the decoder block.
