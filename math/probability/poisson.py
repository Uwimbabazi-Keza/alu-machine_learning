#!/usr/bin/env python3
"""
Creates a class Poisson that represents a poisson distribution
"""


class Poisson:
    """
    Poisson distribution class
    """
    def __init__(self, data=None, lambtha=1.):
        """initialize"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes."
        """
        k = int(k) if not isinstance(k, int) else k
        if k < 0:
            return 0
        """factorial"""
        f = 1
        for i in range(1, k + 1):
            f *= i
        x = self.lambtha ** k * 2.7182818285 ** (-self.lambtha)
        return x / f
