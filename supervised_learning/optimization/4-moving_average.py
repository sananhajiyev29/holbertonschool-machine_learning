#!/usr/bin/env python3
"""Module that calculates the weighted moving average of a data set."""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set.

    Args:
        data: list of data to calculate the moving average of.
        beta: the weight used for the moving average.

    Returns:
        A list containing the moving averages of data.
    """
    v = 0
    averages = []

    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        bias_correction = 1 - beta ** (i + 1)
        averages.append(v / bias_correction)

    return averages
