#!/usr/bin/env python3
"""Module that calculates the cost with L2 regularization in Keras."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: tensor containing the cost without L2 regularization.
        model: a Keras model that includes layers with L2 regularization.

    Returns:
        Tensor containing the total cost for each layer of the network.
    """
    return cost + model.losses
