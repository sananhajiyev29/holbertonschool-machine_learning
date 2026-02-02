    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes

        k: number of successes
        Returns: PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e = 2.7182818285

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        return (e ** (-self.lambtha)) * (self.lambtha ** k) / factorial
