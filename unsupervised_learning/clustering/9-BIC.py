#!/usr/bin/env python3
"""BIC model selection for GMM."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Computes best k using Bayesian Information Criterion."""

    # ------------------------
    # VALIDATION
    # ------------------------
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2):
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (float, int)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    ks = np.arange(kmin, kmax + 1)

    l = np.zeros(len(ks))
    b = np.zeros(len(ks))

    best_bic = np.inf
    best_k = None
    best_result = None

    # ------------------------
    # SINGLE LOOP (REQUIRED)
    # ------------------------
    for i, k in enumerate(ks):

        result = expectation_maximization(
            X, k,
            iterations=iterations,
            tol=float(tol),
            verbose=verbose
        )

        # IMPORTANT: EM failure guard
        if (result is None or result[4] is None):
            l[i] = 0
            b[i] = np.inf
            continue

        pi, m, S, g, log_likelihood = result

        # EXTRA SAFETY
        if log_likelihood is None:
            l[i] = 0
            b[i] = np.inf
            continue

        l[i] = log_likelihood

        # ------------------------
        # BIC computation
        # ------------------------
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)

        bic = p * np.log(n) - 2 * log_likelihood
        b[i] = bic

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
