#!/usr/bin/env python3
"""Slices a DataFrame by selecting specific columns and rows."""


def slice(df):
    """Extracts High, Low, Close, and Volume_(BTC) columns every 60th row."""
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
