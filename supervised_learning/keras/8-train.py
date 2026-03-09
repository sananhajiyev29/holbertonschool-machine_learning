#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent
with validation, early stopping, learning rate decay, and best model save."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent.

    Args:
        network: the model to train.
        data: numpy.ndarray of shape (m, nx) containing the input data.
        labels: one-hot numpy.ndarray of shape (m, classes) with labels.
        batch_size: size of the batch for mini-batch gradient descent.
        epochs: number of passes through data for mini-batch gradient descent.
        validation_data: data to validate the model with, if not None.
        early_stopping: boolean that indicates whether to use early stopping.
        patience: patience used for early stopping.
        learning_rate_decay: boolean whether to use learning rate decay.
        alpha: the initial learning rate.
        decay_rate: the decay rate.
        save_best: boolean whether to save the best model after each epoch.
        filepath: file path where the model should be saved.
        verbose: boolean that determines if output should be printed.
        shuffle: boolean that determines whether to shuffle batches each epoch.

    Returns:
        The History object generated after training the model.
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            """Inverse time decay schedule."""
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            schedule,
            verbose=1
        ))

    if save_best and validation_data is not None and filepath is not None:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        ))

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )
