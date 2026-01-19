#!/usr/bin/env python3
"""
Module that defines a function to add two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise

    Args:
        arr1 (list): First list of numbers
        arr2 (list): Second list of numbers

    Returns:
        list: A new list with element-wise sums
        None: If the two lists are not the same length
    """
    if len(arr1) != len(arr2):
        return None

    return [arr1[i] + arr2[i] for i in range(len(arr1))]
