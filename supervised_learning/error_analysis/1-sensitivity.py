#!/usr/bin/env python3
"""Module for calculating sensitivity from a confusion matrix."""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class."""
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    return true_positives / actual_positives
