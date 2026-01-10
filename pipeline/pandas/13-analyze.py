#!/usr/bin/env python3
"""Compute descriptive statistics for a DataFrame, excluding Timestamp."""


def analyze(df):
    """
    Returns descriptive statistics for all columns except 'Timestamp'.

    Args:
        df: The input DataFrame.

    Returns:
        DataFrame containing count, mean, std, min, 25%, 50%, 75%, max.
    """
    df_stats = df.drop(columns=['Timestamp'], errors='ignore').describe()
    return df_stats
