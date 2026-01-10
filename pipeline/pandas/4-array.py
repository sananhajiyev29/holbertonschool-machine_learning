#!/usr/bin/env python3
"""Converts selected DataFrame values to a NumPy array."""


def array(df):
    """Returns a NumPy array of the last 10 High and Close values."""
    return df[['High', 'Close']].tail(10).to_numpy()
