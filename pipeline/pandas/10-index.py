#!/usr/bin/env python3
"""Sets the Timestamp column as the index of the DataFrame."""


def index(df):
    """
    Sets the Timestamp column as the index.

    Args:
        df (pd.DataFrame): DataFrame with a Timestamp column.

    Returns:
        pd.DataFrame: Modified DataFrame with Timestamp as index.
    """
    df = df.copy()
    df.set_index("Timestamp", inplace=True)
    return df
