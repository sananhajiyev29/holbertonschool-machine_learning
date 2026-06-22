#!/usr/bin/env python3
"""Module that preprocesses raw BTC data for time series forecasting."""
import numpy as np
import pandas as pd


def preprocess(filename):
    """Preprocesses a raw BTC dataset.

    Args:
        filename: path to the raw CSV dataset.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, mean, std).
    """
    df = pd.read_csv(filename)

    # Drop rows with missing values (raw data has NaNs)
    df = df.dropna()

    # Keep only useful columns: Close price and Volume_(BTC)
    # The Close price is the target; we keep a few informative features.
    df = df[['Close', 'Volume_(BTC)', 'Volume_(Currency)',
             'Weighted_Price']]

    # Downsample from 1-minute rows to 1-hour rows: take every 60th row.
    df = df.iloc[::60, :]
    df = df.reset_index(drop=True)

    data = df.values.astype('float32')

    # 70/30 train/validation split
    n = data.shape[0]
    split = int(n * 0.7)

    train = data[:split]
    val = data[split:]

    # Normalize using training statistics only
    mean = train.mean(axis=0)
    std = train.std(axis=0)

    train = (train - mean) / std
    val = (val - mean) / std

    # Build sequences: 24 hours in, predict next hour's close
    window = 24

    def make_sequences(series):
        """Builds (X, y) sliding windows from a normalized series."""
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i + window])
            y.append(series[i + window, 0])
        return np.array(X), np.array(y)

    X_train, y_train = make_sequences(train)
    X_val, y_val = make_sequences(val)

    return X_train, y_train, X_val, y_val, mean, std


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, mean, std = preprocess(
        'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    )
    np.savez(
        'preprocessed_data.npz',
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        mean=mean, std=std
    )
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
