#!/usr/bin/env python3
"""
Task 2: Change of Scale
Plots the exponential decay of C-14 as a line graph with a logarithmic y-axis.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots y = exp((r / t) * x) with a logarithmic y-axis.

    - x-axis: Time (years)
    - y-axis: Fraction Remaining (log scale)
    - Title: Exponential Decay of C-14
    - x-axis range: 0 to 28650
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, color='blue')  # standard line color
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")
    plt.yscale('log')
    plt.xlim(0, 28650)
    plt.show()
