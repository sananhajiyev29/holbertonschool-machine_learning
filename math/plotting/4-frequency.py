#!/usr/bin/env python3
"""
Task 4: Frequency
Plots a histogram of student grades for Project A with specified bins and outlines.
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
    bins = np.arange(0, 101, 10)  # bins every 10 units from 0 to 100
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.show()
