#!/usr/bin/env python3
"""Removes entries with NaN in the Close column of a DataFrame."""


def prune(df):
    """Returns a DataFrame without rows where Close is NaN."""
    return df.dropna(subset=["Close"])
