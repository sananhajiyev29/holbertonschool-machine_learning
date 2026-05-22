#!/usr/bin/env python3
"""Module for BIC model selection for a GMM."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds best number of clusters for GMM using Bayesian Information Criterion.

    Returns:
        best_k, best_result, l, b
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2):
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None

    if (not isinstance(iterations, int) or iterations <= 0):
        return None, None, None, None

    if (not isinstance(tol, float) and not isinstance(tol, int)):
        return None, None, None, None

    if tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    ks = np.arange(kmin, kmax + 1)
    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    best_bic = None
    best_k = None
    best_result = None

    # at most ONE loop as required
    for i, k in enumerate(ks):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        l[i] = log_likelihood

        # number of parameters:
        # pi: k-1 (since sum=1 constraint)
        # means: k*d
        # covariances: k*d*(d+1)/2
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)

        bic = p * np.log(n) - 2 * log_likelihood
        b[i] = bic

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
