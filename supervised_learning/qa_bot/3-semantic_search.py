#!/usr/bin/env python3
"""Module that performs semantic search on a corpus of documents."""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Performs semantic search on a corpus of documents.

    Args:
        corpus_path: path to the corpus of reference documents on which
            to perform semantic search.
        sentence: the sentence from which to perform semantic search.

    Returns:
        The reference text of the document most similar to sentence.
    """
    documents = [sentence]

    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        path = os.path.join(corpus_path, filename)
        with open(path, 'r', encoding='utf-8') as f:
            documents.append(f.read())

    embed = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    )
    embeddings = embed(documents)

    correlation = np.inner(embeddings, embeddings)
    closest = np.argmax(correlation[0, 1:]) + 1

    return documents[closest]
