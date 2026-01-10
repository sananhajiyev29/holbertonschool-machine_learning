#!/usr/bin/env python3
"""
Task 4: Frequency
Plots a histogram of student grades for Project A with bins every 10 units
and bars outlined in black.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades.

    - x-axis: Grades
    - y-axis: Number of Students
    - Bins: every 10 units
    - Title: Project A
    - Bars outlined in black
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    bins = np.arange(0, 110, 10)  # ensure bins cover 0-100 inclusive
    plt.hist(student_grades,
             bins=bins,
             range=(0, 100),
             color='blue',      # bar fill color matches reference
             edgecolor='black') # bar outlines
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.show()
