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
            vari = float((sum(map(lambda n: pow(n - m,
                        2), data)) / len(data)))
            self.p = - (vari / m) + 1
            self.n = round(m / self.p)
            self.p = m / self.n
