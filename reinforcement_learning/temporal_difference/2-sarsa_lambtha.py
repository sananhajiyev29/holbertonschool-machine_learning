#!/usr/bin/env python3
"""Module that performs the SARSA(lambda) algorithm."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action.

    Args:
        Q: numpy.ndarray containing the Q table.
        state: the current state.
        epsilon: the epsilon to use for the calculation.

    Returns:
        The next action index.
    """
    p = np.random.uniform()

    if p > epsilon:
        return np.argmax(Q[state])

    return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Performs the SARSA(lambda) algorithm.

    Args:
        env: the environment instance.
        Q: numpy.ndarray of shape (s, a) containing the Q table.
        lambtha: the eligibility trace factor.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.
        epsilon: the initial threshold for epsilon greedy.
        min_epsilon: the minimum value that epsilon should decay to.
        epsilon_decay: the decay rate for updating epsilon between
            episodes.

    Returns:
        Q, the updated Q table.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = (
                reward + gamma * Q[next_state, next_action] -
                Q[state, action]
            )

            eligibility *= gamma * lambtha
            eligibility[state, action] += 1

            Q += alpha * delta * eligibility

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        epsilon = (
            min_epsilon + (initial_epsilon - min_epsilon) *
            np.exp(-epsilon_decay * episode)
        )

    return Q
