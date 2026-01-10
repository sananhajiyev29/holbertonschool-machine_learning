#!/usr/bin/env python3
"""Fills missing values in a DataFrame according to specific rules."""


def fill(df):
    """
    Modifies a DataFrame by:
    - Removing Weighted_Price column
    - Filling missing Close values with previous row
    - Filling missing Open, High, Low with Close value in same row
    - Filling missing Volume_(BTC) and Volume_(Currency) with 0
    """
    df = df.copy()
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])
    # Fill Close with previous row
    df["Close"] = df["Close"].fillna(method="ffill")
    # Fill Open, High, Low with Close of same row
    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])
    # Fill volumes with 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)
    return df
