#!/usr/bin/env python3
"""Module that determines if gradient descent should stop early."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if gradient descent should stop early.

    Args:
        cost: the current validation cost of the neural network.
        opt_cost: the lowest recorded validation cost of the neural network.
        threshold: the threshold used for early stopping.
        patience: the patience count used for early stopping.
        count: the count of how long the threshold has not been met.

    Returns:
        A boolean of whether the network should be stopped early,
        followed by the updated count.
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count
