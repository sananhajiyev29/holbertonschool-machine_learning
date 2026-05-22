#!/usr/bin/env python3
"""Module that calculates a GMM from a dataset."""
import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: the number of clusters.

    Returns:
        Tuple of (pi, m, S, clss, bic).
    """
    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
