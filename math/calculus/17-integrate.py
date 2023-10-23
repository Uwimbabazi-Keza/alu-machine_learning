#!/usr/bin/env python3
"""
a function def poly_integral(poly, C=0):
that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    def poly_integral(poly, C=0)
    """

    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly) or not isinstance(C, int):
        return None

    integral = [C]

    for power, coeff in enumerate(poly):
        if coeff != 0:
            new_coeff = coeff / (power + 1)
            integral.append(new_coeff)

    return integral
