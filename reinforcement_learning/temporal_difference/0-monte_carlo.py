#!/usr/bin/env python3
"""Module that performs the Monte Carlo algorithm."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Performs the Monte Carlo algorithm.

    Args:
        env: environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: function that takes in a state and returns the next
            action to take.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.

    Returns:
        V, the updated value estimate.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = new_state

        episode_data = np.array(episode_data, dtype=int)
        G = 0

        for i in reversed(range(len(episode_data))):
            state, reward = episode_data[i]
            G = gamma * G + reward
            V[state] = V[state] + alpha * (G - V[state])

    return V
