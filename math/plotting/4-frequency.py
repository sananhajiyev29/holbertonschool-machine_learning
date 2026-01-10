#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    """Plot a histogram of student grades for Project A."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    bins = range(0, 101, 10)  # bins every 10 units
    plt.hist(student_grades, bins=bins, edgecolor='black')

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    # Do not call plt.show() for autograder
