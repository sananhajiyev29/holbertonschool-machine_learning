#!/usr/bin/env python3
"""Concatenate DataFrames with Timestamp as first level in MultiIndex."""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Index both DataFrames on Timestamp, select range of timestamps,
    concatenate with keys, and sort the result chronologically.
    """
    df1 = index(df1)
    df2 = index(df2)

    df1_subset = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2_subset = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]

    df_concat = pd.concat(
        [df2_subset, df1_subset], keys=["bitstamp", "coinbase"]
    )
    df_concat = df_concat.reorder_levels([1, 0]).sort_index()
    return df_concat
