#!/usr/bin/env python3
"""
Task 7: Gradient
Creates a scatter plot of sampled elevations on a mountain with colorbar.
"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    Creates a scatter plot of mountain elevation data.

    Plots x,y coordinates colored by elevation (z) values.
    Includes a colorbar to show the elevation scale in meters.
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    scatter = plt.scatter(x, y, c=z)
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')
    cbar = plt.colorbar(scatter)
    cbar.set_label('elevation (m)')
    plt.show()
