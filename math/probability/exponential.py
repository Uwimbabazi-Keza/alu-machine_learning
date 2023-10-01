#!/usr/bin/env python3
"""
Create a class Exponential that represents an exponential distribution
"""


class Exponential:
    """
    Exponential distribution class
    """

    def __init__(self, data=None, lambtha=1.):
        """
        initializes
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Update the class Exponential
        """
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285 ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Update the class Exponential:
        Calculates the value of the CDF for a given time period
        """
        if x < 0:
            return 0
        return 1 - 2.7182818285 ** (-self.lambtha * x)
