#!/usr/bin/env python3
"""
17-integrate.py
Calculate the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial represented by a list of coefficients.

    Args:
        poly (list): Coefficients of the polynomial, index = power of x
        C (int): Integration constant

    Returns:
        list: Coefficients of the integral polynomial
    """
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, int):
        return None

    integral = [C]
    for i, coeff in enumerate(poly):
        val = coeff / (i + 1)
        # Represent as int if whole number
        if val.is_integer():
            val = int(val)
        integral.append(val)

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
