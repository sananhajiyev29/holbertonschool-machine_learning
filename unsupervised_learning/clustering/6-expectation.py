#!/usr/bin/env python3
"""Module for calculating the expectation step in a GMM."""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        pi: numpy.ndarray of shape (k,) containing cluster priors.
        m: numpy.ndarray of shape (k, d) containing centroid means.
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices.

    Returns:
        g, l, or None, None on failure.
        g is a numpy.ndarray of shape (k, n) containing posteriors.
        l is the total log likelihood.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(pi, np.ndarray) or len(pi.shape) != 1 or
            not isinstance(m, np.ndarray) or len(m.shape) != 2 or
            not isinstance(S, np.ndarray) or len(S.shape) != 3):
        return None, None

    n, d = X.shape
    k = pi.shape[0]
    if (k == 0 or m.shape != (k, d) or S.shape != (k, d, d) or
            np.any(pi < 0) or not np.isclose(np.sum(pi), 1)):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    likelihoods = np.sum(g, axis=0)
    if np.any(likelihoods <= 0):
        return None, None

    log_likelihood = np.sum(np.log(likelihoods))
    g = g / likelihoods

    return g, log_likelihood
