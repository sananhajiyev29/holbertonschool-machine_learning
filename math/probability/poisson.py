#!/usr/bin/env python3
"""
Poisson distribution module
"""


class Poisson:
    """
    Represents a Poisson distribution
    """

        def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        cdf = 0
        for i in range(0, k + 1):
            cdf += self.pmf(i)

        return cdf
