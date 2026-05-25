#!/usr/bin/env python3
"""Module that represents a noiseless 1D Gaussian process."""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initializes the GaussianProcess class.

        Args:
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                already sampled with the black-box function.
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                of the black-box function for each input in X_init.
            l: the length parameter for the kernel.
            sigma_f: the standard deviation given to the output of the
                black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices.

        Args:
            X1: numpy.ndarray of shape (m, 1).
            X2: numpy.ndarray of shape (n, 1).

        Returns:
            The covariance kernel matrix as a numpy.ndarray of shape (m, n).
        """
        sqdist = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1) +
            np.sum(X2 ** 2, axis=1) -
            2 * np.dot(X1, X2.T)
        )
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """Predicts the mean and standard deviation of points in the GP.

        Args:
            X_s: numpy.ndarray of shape (s, 1) containing all points
                whose mean and standard deviation should be calculated.

        Returns:
            mu: numpy.ndarray of shape (s,) containing the mean.
            sigma: numpy.ndarray of shape (s,) containing the variance.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu, sigma
