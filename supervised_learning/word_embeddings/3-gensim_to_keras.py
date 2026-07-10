#!/usr/bin/env python3
"""Module that converts a gensim word2vec model to a keras layer."""
import tensorflow as tf


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: a trained gensim word2vec model.

    Returns:
        The trainable keras Embedding layer.
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return layer
