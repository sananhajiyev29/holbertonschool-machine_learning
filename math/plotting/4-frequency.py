#!/usr/bin/env python3
"""
Task 4: Frequency
Plots a histogram of student grades for Project A.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades for Project A.
    
    Creates a histogram showing the distribution of 50 student grades
    with bins every 10 units from 0 to 100, outlined in black.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    
    plt.show()
