#!/usr/bin/env python3
"""
Create a class Binomial that represents a binomial distribution
"""


class Binomial:
    """
    Initialize Binomial
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        initialize
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            m = float(sum(data) / len(data))
            vari = float((sum(map(lambda n: pow(n - m, 2), data)) / len(data)))
            self.p = - (vari / m) + 1
            self.n = round(m / self.p)
            self.p = m / self.n

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes"
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        binomial_coefficient = 1
        for i in range(1, k + 1):
            binomial_coefficient *= (self.n - i + 1) // i
        pmf_value = binomial_coefficient * (self.p ** k) * ((1 - self.p)
                    ** (self.n - k))

        return pmf_value
