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
            hidden: the number of hidden units in the fully connected layer.
            drop_rate: the dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Performs the forward pass of the decoder block.

        Args:
            x: tensor of shape (batch, target_seq_len, dm).
            encoder_output: tensor of shape
                (batch, input_seq_len, dm).
            training: boolean indicating training mode.
            look_ahead_mask: mask for the first MHA.
            padding_mask: mask for the second MHA.

        Returns:
            Tensor of shape (batch, target_seq_len, dm).
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(
            out1,
            encoder_output,
            encoder_output,
            padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        hidden = self.dense_hidden(out2)
        output = self.dense_output(hidden)
        output = self.dropout3(output, training=training)
        out3 = self.layernorm3(out2 + output)

        return out3
