#!/usr/bin/env python3
"""Module that creates a bag of words embedding matrix."""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of the vocabulary words to use for the analysis.
            If None, all words within sentences are used.

    Returns:
        Tuple of (embeddings, features) where embeddings is a
        numpy.ndarray of shape (s, f) containing the embeddings and
        features is a list of the features used.
    """
    tokenized = []
    for sentence in sentences:
        words = re.findall(r"\w+", sentence.lower())
        words = [w for w in words if not (w == "s" and len(w) == 1)]
        tokenized.append(words)

    if vocab is None:
        all_words = set()
        for words in tokenized:
            for word in words:
                all_words.add(word)
        features = sorted(all_words)
    else:
        features = list(vocab)

    word_index = {word: i for i, word in enumerate(features)}

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, words in enumerate(tokenized):
        for word in words:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, np.array(features)
