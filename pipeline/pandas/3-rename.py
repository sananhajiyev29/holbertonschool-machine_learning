#!/usr/bin/env python3
"""Renames Timestamp column and converts it to datetime."""

import pandas as pd


def rename(df):
    """Renames Timestamp to Datetime and returns selected columns."""
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
