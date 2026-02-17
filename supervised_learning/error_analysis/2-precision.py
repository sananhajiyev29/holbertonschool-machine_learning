#!/usr/bin/env python3
"""Module for calculating precision from a confusion matrix."""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class."""
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return true_positives / predicted_positives
