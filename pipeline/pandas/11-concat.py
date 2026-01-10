#!/usr/bin/env python3
"""Concatenate two DataFrames with Timestamp indexing and keys."""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Index both DataFrames on Timestamp, select subset of df2,
    and concatenate on top of df1 with keys.
    """
    df1 = index(df1)
    df2 = index(df2)
    df2_subset = df2[df2.index <= 1417411920]
    df_concat = pd.concat([df2_subset, df1], keys=["bitstamp", "coinbase"])
    return df_concat
