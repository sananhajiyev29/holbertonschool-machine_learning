#!/usr/bin/env python3
"""
Creates a pandas DataFrame from a NumPy ndarray.
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pandas DataFrame from a NumPy ndarray.

    Args:
        array (np.ndarray): The NumPy array to convert.

    Returns:
        pd.DataFrame: DataFrame with alphabetically labeled columns.
    """
    columns = [chr(ord('A') + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
