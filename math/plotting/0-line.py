#!/usr/bin/env python3
"""
0. Line Graph
Plot y = x^3 as a solid red line from x=0 to x=10.
Uses numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plot a cubic line graph.
    - x-axis ranges from 0 to 10
    - y = x^3
    - line is solid red
    """
    x = np.arange(0, 11)
    y = x ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, 'r-', label='y = x^3')  # solid red line
    plt.xlim(0, 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Line Graph of y = x^3")
    plt.legend()
    plt.show()
