#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    """Plot a histogram of student grades for Project A."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # bins every 10 units from 0 to 100
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # histogram with black edges
    plt.hist(student_grades, bins=bins, edgecolor='black')
    
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    # Do not call plt.show() for autograder

    plt.show()
