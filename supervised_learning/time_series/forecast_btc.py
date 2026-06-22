#!/usr/bin/env python3
"""Module that creates, trains, and validates a Keras model for BTC."""
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_preprocessed(path='preprocessed_data.npz'):
    """Loads the preprocessed data.

    Args:
        path: path to the .npz file with preprocessed data.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val).
    """
    data = np.load(path)
    return (data['X_train'], data['y_train'],
            data['X_val'], data['y_val'])


def make_dataset(X, y, batch_size=256, shuffle=False):
    """Builds a tf.data.Dataset from arrays.

    Args:
        X: input sequences.
        y: target values.
        batch_size: the batch size.
        shuffle: whether to shuffle the data.

    Returns:
        A batched tf.data.Dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape):
    """Builds the RNN forecasting model.

    Args:
        input_shape: shape of a single input sequence (window, features).

    Returns:
        A compiled Keras model.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
    X_train, y_train, X_val, y_val = load_preprocessed()

    train_ds = make_dataset(X_train, y_train, shuffle=True)
    val_ds = make_dataset(X_val, y_val)

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stop]
    )

    val_loss = model.evaluate(val_ds)
    print("Validation MSE:", val_loss)

    model.save('btc_forecast_model.h5')
