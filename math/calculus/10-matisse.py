#!/usr/bin/env python3
"""Calculate the derivative of a polynomial represented as a list of coefficients."""


def poly_derivative(poly):
    """Return the derivative of a polynomial as a list of coefficients."""
    if not isinstance(poly, list) or not poly or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if len(poly) == 1:  # derivative of constant is 0
        return [0]
    # derivative: multiply each coefficient by its power (skip constant at index 0)
    derivative = [i * poly[i] for i in range(1, len(poly))]
    return derivative if derivative else [0]
