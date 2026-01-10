#!/usr/bin/env python3
"""Sorts a DataFrame in reverse chronological order and transposes it."""


def flip_switch(df):
    """Returns a DataFrame sorted in reverse order and transposed."""
    return df.iloc[::-1].transpose()
