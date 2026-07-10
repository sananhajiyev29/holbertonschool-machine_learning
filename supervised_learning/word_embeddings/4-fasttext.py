#!/usr/bin/env python3
"""Module that creates, builds, and trains a gensim fastText model."""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a gensim fastText model.

    Args:
        sentences: list of sentences to be trained on.
        vector_size: dimensionality of the embedding layer.
        min_count: minimum number of occurrences of a word for training.
        negative: size of negative sampling.
        window: maximum distance between the current and predicted word.
        cbow: boolean determining training type; True for CBOW,
            False for Skip-gram.
        epochs: number of iterations to train over.
        seed: seed for the random number generator.
        workers: number of worker threads to train the model.

    Returns:
        The trained model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
        epochs=epochs
    )

    return model
