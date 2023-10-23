#!/usr/bin/env python3
"""
function def poly_derivative(poly):
that calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    def poly_derivative(poly)
    """
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    derivative = [coeff * power for power, coeff in enumerate(poly[1:], start=1) if coeff != 0]

    if not derivative:
        return [0]

    return derivative
