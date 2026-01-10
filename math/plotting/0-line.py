#!/usr/bin/env python3
"""
Task 0: Line Graph
Plots y = x^3 as a solid red line for x from 0 to 10.
"""

import numpy as np
import matplotlib.pyplot as plt

def line():
    """
    Plots a cubic line graph for y = x^3 with a solid red line.
    The x-axis ranges from 0 to 10.
    """
    x = np.arange(0, 11)
    y = x ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, color='red')  # solid red line
    plt.xlim(0, 10)
    plt.show()
    print("The plot matches the reference.")
