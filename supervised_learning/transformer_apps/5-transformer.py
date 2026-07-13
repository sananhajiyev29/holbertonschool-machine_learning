#!/usr/bin/env python3
"""Module that creates a transformer network."""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args:
        max_seq_len: integer representing the maximum sequence length.
        dm: the model depth.

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors.
    """
    PE = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention.

    Args:
        Q: query matrix tensor.
        K: key matrix tensor.
        V: value matrix tensor.
        mask: optional mask tensor.

    Returns:
        Tuple of (output, weights).
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled += mask * -1e9

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention."""

    def __init__(self, dm, h):
        """Initializes the MultiHeadAttention layer.

        Args:
            dm: dimensionality of the model.
            h: number of heads.
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
            The reshaped and transposed tensor.
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Performs multi head attention.

        Args:
            Q: query input tensor.
            K: key input tensor.
            V: value input tensor.
            mask: the mask to apply.

        Returns:
            Tuple of (output, weights).
        """
        batch = tf.shape(Q)[0]

        Q = self.split_heads(self.Wq(Q), batch)
        K = self.split_heads(self.Wk(K), batch)
        V = self.split_heads(self.Wv(V), batch)

        attention, weights = sdp_attention(Q, K, V, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(attention, (batch, -1, self.dm))
        output = self.linear(concat)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """Creates an encoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the EncoderBlock.

        Args:
            dm: dimensionality of the model.
            h: number of heads.
            hidden: number of hidden units in the fully connected layer.
            drop_rate: the dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Performs the forward pass of the encoder block.

        Args:
            x: input tensor.
            training: boolean for training mode.
            mask: the mask to apply.

        Returns:
            The block's output tensor.
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)

        hidden = self.dense_hidden(out1)
        output = self.dense_output(hidden)
        output = self.dropout2(output, training=training)
        out2 = self.layernorm2(out1 + output)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """Creates a decoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the DecoderBlock.

        Args:
            dm: dimensionality of the model.
            h: number of heads.
            hidden: number of hidden units in the fully connected layer.
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
            x: input tensor.
            encoder_output: output of the encoder.
            training: boolean for training mode.
            look_ahead_mask: mask for the first attention layer.
            padding_mask: mask for the second attention layer.

        Returns:
            The block's output tensor.
        """
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
        )
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(out1 + attention2)

        hidden = self.dense_hidden(out2)
        output = self.dense_output(hidden)
        output = self.dropout3(output, training=training)
        out3 = self.layernorm3(out2 + output)

        return out3


class Encoder(tf.keras.layers.Layer):
    """Creates the encoder for a transformer."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initializes the Encoder.

        Args:
            N: number of blocks in the encoder.
            dm: dimensionality of the model.
            h: number of heads.
            hidden: number of hidden units in the fully connected layer.
            input_vocab: size of the input vocabulary.
            max_seq_len: maximum sequence length possible.
            drop_rate: the dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Performs the forward pass of the encoder.

        Args:
            x: input tensor.
            training: boolean for training mode.
            mask: the mask to apply.

        Returns:
            The encoder's output tensor.
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Creates the decoder for a transformer."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initializes the Decoder.

        Args:
            N: number of blocks in the decoder.
            dm: dimensionality of the model.
            h: number of heads.
            hidden: number of hidden units in the fully connected layer.
            target_vocab: size of the target vocabulary.
            max_seq_len: maximum sequence length possible.
            drop_rate: the dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Performs the forward pass of the decoder.

        Args:
            x: input tensor.
            encoder_output: output of the encoder.
            training: boolean for training mode.
            look_ahead_mask: mask for the first attention layer.
            padding_mask: mask for the second attention layer.

        Returns:
            The decoder's output tensor.
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](
                x, encoder_output, training, look_ahead_mask, padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """Creates a transformer network."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initializes the Transformer.

        Args:
            N: number of blocks in the encoder and decoder.
            dm: dimensionality of the model.
            h: number of heads.
            hidden: number of hidden units in the fully connected layers.
            input_vocab: size of the input vocabulary.
            target_vocab: size of the target vocabulary.
            max_seq_input: maximum sequence length possible for the input.
            max_seq_target: maximum sequence length possible for the
                target.
            drop_rate: the dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """Performs the forward pass of the transformer.

        Args:
            inputs: input tensor.
            target: target tensor.
            training: boolean for training mode.
            encoder_mask: padding mask for the encoder.
            look_ahead_mask: look ahead mask for the decoder.
            decoder_mask: padding mask for the decoder.

        Returns:
            The transformer's output tensor.
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(
            target, encoder_output, training, look_ahead_mask, decoder_mask
        )
        output = self.linear(decoder_output)

        return output
