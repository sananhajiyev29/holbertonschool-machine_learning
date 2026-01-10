#!/usr/bin/env python3
"""
Task 1: Scatter Plot
Plots Men's Height vs Weight as a scatter plot with magenta points.
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Generates a scatter plot of height vs weight.

    - x-axis: Height (in)
    - y-axis: Weight (lbs)
    - Title: Men's Height vs Weight
    - Data points: magenta
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, color='magenta')
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.title("Men's Height vs Weight")
    plt.show()
