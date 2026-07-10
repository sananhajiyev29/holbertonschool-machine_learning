#!/usr/bin/env python3
"""Module that creates a TF-IDF embedding matrix."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of the vocabulary words to use for the analysis.
            If None, all words within sentences are used.

    Returns:
        Tuple of (embeddings, features) where embeddings is a
        numpy.ndarray of shape (s, f) containing the embeddings and
        features is a list of the features used.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)

    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
