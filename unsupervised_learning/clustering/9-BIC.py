#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM model selection."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using the
    Bayesian Information Criterion.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive int, minimum number of clusters to check (inclusive)
        kmax: positive int, maximum number of clusters to check (inclusive)
        iterations: positive int, max iterations for the EM algorithm
        tol: non-negative float, tolerance for the EM algorithm
        verbose: bool, whether EM should print information

    Returns:
        best_k, best_result, l, b, or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmax > n:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    k_range = kmax - kmin + 1
    log_likelihoods = np.empty(k_range)
    bics = np.empty(k_range)
    results = []
    best_k = None
    best_result = None
    best_bic = None

    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, _, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        # Number of parameters:
        # (k - 1) priors + k*d means + k * d*(d+1)/2 covariance entries
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)
        bic = p * np.log(n) - 2 * log_l

        log_likelihoods[idx] = log_l
        bics[idx] = bic
        results.append((pi, m, S))

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics
