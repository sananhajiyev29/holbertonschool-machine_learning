#!/usr/bin/env python3
"""Module that performs Bayesian optimization on a 1D Gaussian process."""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initializes the BayesianOptimization class.

        Args:
            f: the black-box function to be optimized.
            X_init: numpy.ndarray of shape (t, 1) representing inputs
                already sampled with the black-box function.
            Y_init: numpy.ndarray of shape (t, 1) representing outputs.
            bounds: tuple of (min, max) representing the bounds of the
                space in which to look for the optimal point.
            ac_samples: number of samples that should be analyzed.
            l: the length parameter for the kernel.
            sigma_f: the standard deviation given to the output of the
                black-box function.
            xsi: the exploration-exploitation factor for acquisition.
            minimize: bool determining whether optimization is for
                minimization (True) or maximization (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        min_b, max_b = bounds
        self.X_s = np.linspace(min_b, max_b, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location using Expected
        Improvement.

        Returns:
            X_next: numpy.ndarray of shape (1,) with the next best
                sample point.
            EI: numpy.ndarray of shape (ac_samples,) with the expected
                improvement of each potential sample.
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros(sigma.shape)
            for i in range(sigma.shape[0]):
                if sigma[i] > 0:
                    Z[i] = imp[i] / sigma[i]
                else:
                    Z[i] = 0
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function.

        Args:
            iterations: the maximum number of iterations to perform.

        Returns:
            X_opt: numpy.ndarray of shape (1,) representing the optimal
                point.
            Y_opt: numpy.ndarray of shape (1,) representing the optimal
                function value.
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.all(self.gp.X == X_next, axis=1)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
