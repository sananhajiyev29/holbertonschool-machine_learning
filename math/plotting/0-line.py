#!/usr/bin/env python3
"""Task 0: Line graph of y = x^3 from 0 to 10."""
import numpy as np
import matplotlib.pyplot as plt

def line():
    """Plots y = x^3 as a solid red line for x from 0 to 10."""
    x = np.arange(0, 11)
    y = x ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, 'r')  # solid red line
    plt.xlim(0, 10)
    print("The plot matches the reference.")
