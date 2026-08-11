#!/usr/bin/env python3
"""Module that has the trained agent play an episode."""
import numpy as np


def play(env, Q, max_steps=100):
    """Has the trained agent play an episode.

    Args:
        env: the FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        max_steps: the maximum number of steps in the episode.

    Returns:
        Tuple of (total_rewards, rendered_outputs) where total_rewards
        is the total reward for the episode and rendered_outputs is a
        list of rendered board states at each step.
    """
    state, _ = env.reset()
    rendered_outputs = []
    total_rewards = 0

    for step in range(max_steps):
        rendered_outputs.append(env.render())

        action = np.argmax(Q[state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_rewards += reward
        state = new_state

        if done:
            rendered_outputs.append(env.render())
            break

    return total_rewards, rendered_outputs
