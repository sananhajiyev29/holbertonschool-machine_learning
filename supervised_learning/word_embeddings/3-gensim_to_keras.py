#!/usr/bin/env python3
"""Module that converts a gensim word2vec model to a keras layer."""


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: a trained gensim word2vec model.

    Returns:
        The trainable keras Embedding layer.
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
