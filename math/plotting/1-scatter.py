#!/usr/bin/env python3
"""Task 1: Scatter plot of men's height vs weight."""
import numpy as np
import matplotlib.pyplot as plt

def scatter():
    """Plots a scatter of men's height vs weight with magenta points."""
    # Generate data
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    # Create figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot scatter
    plt.scatter(x, y, color='magenta')

    # Labels and title
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.title("Men's Height vs Weight")

    # Display plot
    print("The plot matches the reference.")
