#!/usr/bin/env python3
"""Module that performs the TD(lambda) algorithm."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm.

    Args:
        env: the environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: function that takes in a state and returns the next
            action to take.
        lambtha: the eligibility trace factor.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.

    Returns:
        V, the updated value estimate.
    """
    for episode in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]

            eligibility *= gamma * lambtha
            eligibility[state] += 1

            V += alpha * delta * eligibility

            if terminated or truncated:
                break

            state = next_state

    return V
