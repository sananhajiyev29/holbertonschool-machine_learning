#!/usr/bin/env python3
"""
0. Line Graph
This module plots a line graph of y = x^3 from x = 0 to 10 using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

def line():
    """
    Plots y = x^3 as a solid red line from x = 0 to 10.
    The plot is labeled and displayed.
    """
    # Create x values
    x = np.arange(0, 11)

    # Compute y values
    y = x ** 3

    # Create figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot y as solid red line
    plt.plot(x, y, 'r-', label='y = x^3')

    # Label axes
    plt.xlabel('x')
    plt.ylabel('y')

    # Add title
    plt.title('Line Graph of y = x^3')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()
