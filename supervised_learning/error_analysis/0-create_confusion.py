#!/usr/bin/env python3
"""Module for creating a confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix from one-hot labels and logits."""
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    np.add.at(confusion, (y_true, y_pred), 1)
    return confusion
