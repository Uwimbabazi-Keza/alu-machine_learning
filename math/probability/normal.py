#!/usr/bin/env python3
"""
Create a class Normal that represents a normal distribution
"""


class Normal:
    """
    Normal distribution class
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        initialize
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = (sum([(x - self.mean) ** 2 for x in data])
                           / len(data)) ** 0.5

    def z_score(self, x):
        """
        z-score
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        x-value
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        PDF
        """
        p = 3.1415926536
        e = 2.7182818285
        pdf = (e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
               / (self.stddev * ((2 * p) ** 0.5)))
        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        cdf = (0.5 * (1 + self._erf((x - self.mean)
               / (self.stddev * 2 ** 0.5))))
        return cdf

    def _erf(self, z):
        """
        Error function approximation using the Maclaurin series
        """
        result = (2 / (3.1415926536 ** 0.5)) * (z - (z ** 3)
                 / 3 + (z ** 5) / 10 - (z ** 7) / 42 + (z ** 9)
                 / 216)
        return result
