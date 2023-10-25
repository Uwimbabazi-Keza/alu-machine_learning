#!/usr/bin/env python3
"""
Create a class Normal that represents a normal distribution
"""
from math import erf, sqrt, pi, exp


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
        calculates z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        calculates x-value of a given z-score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        calculates the value of the PDF for a given x-value
        """
        p = 3.1415926536
        y = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        pdf = ((1 / (self.stddev * (2 * p) ** 0.5)) * 2.7182818285 ** (y))
        return pdf

    def cdf(self, x):
        """
        calculates the value of the CDF for a given x-value
        """
        z = (x - self.mean) / (self.stddev * sqrt(2))
        cdf = 0.5 * (1 + erf(z))
        return cdf