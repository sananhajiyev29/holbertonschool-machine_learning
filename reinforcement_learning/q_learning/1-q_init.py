#!/usr/bin/env python3
"""Module that initializes the Q-table."""
import numpy as np


def q_init(env):
    """Initializes the Q-table.

    Args:
        env: the FrozenLakeEnv instance.

    Returns:
        The Q-table as a numpy.ndarray of zeros.
    """
    state_space = env.observation_space.n
    action_space = env.action_space.n

    return np.zeros((state_space, action_space))
