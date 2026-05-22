#!/usr/bin/env python3
"""Module that performs a t-SNE transformation."""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """Performs a t-SNE transformation.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        ndims: new dimensional representation of X.
        idims: intermediate dimensional representation after PCA.
        perplexity: the perplexity.
        iterations: the number of iterations.
        lr: the learning rate.

    Returns:
        Y: numpy.ndarray of shape (n, ndim) containing the optimized
            low dimensional transformation of X.
    """
    X = pca(X, idims)
    n, d = X.shape

    P = P_affinities(X, perplexity=perplexity)
    P = P * 4

    Y = np.random.randn(n, ndims)
    Y_prev = np.zeros((n, ndims))

    for i in range(1, iterations + 1):
        dY, Q = grads(Y, P)

        if i < 20:
            momentum = 0.5
        else:
            momentum = 0.8

        Y_new = Y - lr * dY + momentum * (Y - Y_prev)
        Y_prev = Y
        Y = Y_new

        Y = Y - np.mean(Y, axis=0)

        if i == 100:
            P = P / 4

        if i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))

    return Y
