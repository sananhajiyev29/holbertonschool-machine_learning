#!/usr/bin/env python3
"""Script that optimizes a machine learning model using GPyOpt."""
import numpy as np
import GPy
import GPyOpt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as K


def load_data():
    """Loads and preprocesses the MNIST dataset."""
    (X_train, Y_train), (X_test, Y_test) = K.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
    Y_train = K.utils.to_categorical(Y_train, 10)
    Y_test = K.utils.to_categorical(Y_test, 10)
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_data()


def model_fn(params):
    """Builds, trains, and evaluates a model with given hyperparameters."""
    params = params[0]
    lr = float(params[0])
    units = int(params[1])
    dropout = float(params[2])
    l2 = float(params[3])
    batch_size = int(params[4])

    fname = (
        "checkpoint_lr{}_units{}_dropout{}_l2{}_bs{}.h5"
    ).format(lr, units, dropout, l2, batch_size)

    model = K.Sequential([
        K.layers.Dense(
            units, activation='relu',
            kernel_regularizer=K.regularizers.l2(l2),
            input_shape=(784,)
        ),
        K.layers.Dropout(dropout),
        K.layers.Dense(
            units, activation='relu',
            kernel_regularizer=K.regularizers.l2(l2)
        ),
        K.layers.Dropout(dropout),
        K.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = K.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    checkpoint = K.callbacks.ModelCheckpoint(
        fname, monitor='val_loss', save_best_only=True
    )

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=20,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=0
    )

    best_val_loss = min(history.history['val_loss'])
    return best_val_loss


bounds = [
    {'name': 'learning_rate', 'type': 'continuous',
     'domain': (1e-4, 1e-2)},
    {'name': 'units', 'type': 'discrete',
     'domain': (32, 64, 128, 256, 512)},
    {'name': 'dropout', 'type': 'continuous',
     'domain': (0.0, 0.5)},
    {'name': 'l2', 'type': 'continuous',
     'domain': (1e-6, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete',
     'domain': (32, 64, 128, 256)}
]

optimizer = GPyOpt.methods.BayesianOptimization(
    f=model_fn,
    domain=bounds,
    acquisition_type='EI',
    maximize=False
)

optimizer.run_optimization(max_iter=30)

optimizer.plot_convergence()

with open('bayes_opt.txt', 'w') as f:
    f.write("Bayesian Optimization Report\n")
    f.write("=" * 40 + "\n\n")
    f.write("Best hyperparameters found:\n")
    for i, b in enumerate(bounds):
        f.write("  {}: {}\n".format(b['name'], optimizer.x_opt[i]))
    f.write("\nBest validation loss: {}\n".format(optimizer.fx_opt))
    f.write("\nAll evaluations:\n")
    for i, (x, y) in enumerate(zip(optimizer.X, optimizer.Y)):
        f.write("Iteration {}: x={}, y={}\n".format(i, x, y[0]))
