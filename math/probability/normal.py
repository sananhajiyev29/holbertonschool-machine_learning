#!/usr/bin/env python3
"""
Normal distribution module
"""


class Normal:
    """
    Represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes a Normal distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            mean = sum(data) / n
            variance = sum((x - mean) ** 2 for x in data) / n

            self.mean = float(mean)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        pi = 3.1415926536
        e = 2.7182818285

        coeff = 1 / ((2 * pi * (self.stddev ** 2)) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))

        return coeff * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # erf(z) approximation using series expansion
        erf_sum = 0
        factorial = 1
        z_power = z  # z^(2n+1) starts with n=0 -> z^1

        for n in range(0, 21):
            if n > 0:
                factorial *= n
                z_power *= z * z

            term = ((-1) ** n) * z_power / (factorial * (2 * n + 1))
            erf_sum += term

        erf = (2 / (pi ** 0.5)) * erf_sum

        return 0.5 * (1 + erf)
