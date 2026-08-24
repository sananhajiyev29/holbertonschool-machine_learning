#!/usr/bin/env python3
"""Module that implements full training with policy gradients."""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Implements a full training using policy gradients.

    Args:
        env: the initial environment.
        nb_episodes: number of episodes used for training.
        alpha: the learning rate.
        gamma: the discount factor.
        show_result: if True, render the environment every 1000
            episodes.

    Returns:
        All values of the score (sum of all rewards during one episode).
    """
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    weight = np.random.rand(n_states, n_actions)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        gradients = []
        rewards = []
        score = 0

        while True:
            if show_result and episode % 1000 == 0:
                env.render()

            action, gradient = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)

            gradients.append(gradient)
            rewards.append(reward)
            score += reward

            if terminated or truncated:
                break

            state = next_state

        for i in range(len(gradients)):
            G = sum(
                r * (gamma ** t) for t, r in enumerate(rewards[i:])
            )
            weight += alpha * gradients[i] * G

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
