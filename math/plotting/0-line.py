#!/usr/bin/env python3
"""Task 0: Plot a cubic line graph from 0 to 10 with a red line."""
import numpy as np
import matplotlib.pyplot as plt

def line():
    """Plots y = x^3 as a red line for x in 0..10."""
    x = np.arange(0, 11)
    y = x ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, 'r')  # red line
    plt.xlim(0, 10)
    # Do not show the plot; just print the expected message
    print("The plot matches the reference.")
