#!/usr/bin/env python3
"""
Module that performs element-wise operations using numpy.
"""

import numpy as np


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division on numpy arrays.
    """
    return (mat1 + mat2,
            mat1 - mat2,
            mat1 * mat2,
            mat1 / mat2)
