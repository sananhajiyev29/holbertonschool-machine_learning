#!/usr/bin/env python3
"""Module that initializes variables for t-SNE."""
import numpy as np


def P_init(X, perplexity):
    """Initializes variables required to calculate the P affinities in t-SNE.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        perplexity: the perplexity that all Gaussian distributions should
            have.

    Returns:
        Tuple of (D, P, betas, H) where D is squared pairwise distances,
        P is initialized affinities, betas are initialized beta values,
        H is the Shannon entropy for perplexity with base 2.
    """
    n, d = X.shape

    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
