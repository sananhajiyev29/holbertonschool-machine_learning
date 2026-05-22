#!/usr/bin/env python3
"""Expectation Maximization algorithm for a Gaussian Mixture Model."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Perform the expectation maximization for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive int, number of clusters
        iterations: positive int, max iterations for the algorithm
        tol: non-negative float, tolerance of the log likelihood
        verbose: bool, whether to print log likelihood info every 10 iters

    Returns:
        pi, m, S, g, log_l, or None, None, None, None, None on failure
            pi: numpy.ndarray of shape (k,) priors for each cluster
            m: numpy.ndarray of shape (k, d) centroid means
            S: numpy.ndarray of shape (k, d, d) covariance matrices
            g: numpy.ndarray of shape (k, n) posterior probabilities
            log_l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    g, log_l = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {}".format(
            round(log_l, 5)))

    prev_log_l = log_l
    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        g, log_l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(log_l, 5)))

        if abs(log_l - prev_log_l) <= tol:
            break
        prev_log_l = log_l

    if verbose and (i % 10 != 0 or i == iterations):
        print("Log Likelihood after {} iterations: {}".format(
            i, round(log_l, 5)))

    return pi, m, S, g, log_l
