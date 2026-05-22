#!/usr/bin/env python3
"""Module that calculates symmetric P affinities for t-SNE."""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Calculates the symmetric P affinities of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        tol: maximum tolerance allowed for the difference in Shannon
            entropy from perplexity.
        perplexity: the perplexity that all Gaussian distributions
            should have.

    Returns:
        P: numpy.ndarray of shape (n, n) containing the symmetric P
            affinities.
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        low = None
        high = None
        beta = betas[i, 0]

        Di = np.delete(D[i], i)
        Hi, Pi = HP(Di, beta)
        Hdiff = Hi - H

        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                low = beta
                if high is None:
                    beta = beta * 2
                else:
                    beta = (beta + high) / 2
            else:
                high = beta
                if low is None:
                    beta = beta / 2
                else:
                    beta = (beta + low) / 2

            Hi, Pi = HP(Di, beta)
            Hdiff = Hi - H
            tries += 1

        betas[i, 0] = beta
        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    P = (P + P.T) / (2 * n)

    return P
