#!/usr/bin/env python3
"""Sorts a DataFrame by the High column in descending order."""


def high(df):
    """Returns the DataFrame sorted by the High price descending."""
    return df.sort_values(by="High", ascending=False)
