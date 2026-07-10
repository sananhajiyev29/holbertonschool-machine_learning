#!/usr/bin/env python3
"""Module that calculates the positional encoding for a transformer."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args:
        max_seq_len: integer representing the maximum sequence length.
        dm: the model depth.

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors.
    """
    PE = np.zeros((max_seq_len, dm))

    position = np.arange(max_seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
