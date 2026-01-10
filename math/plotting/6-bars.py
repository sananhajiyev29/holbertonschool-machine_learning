#!/usr/bin/env python3
"""
Task 6: Stacking Bars
Plots a stacked bar graph showing fruit quantities per person.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph of fruit quantities per person.
    
    Creates a stacked bar chart where:
    - Each bar represents a person (Farrah, Fred, Felicia)
    - Each segment represents a fruit type (apples, bananas, oranges, peaches)
    - Bars are stacked from bottom to top in the order of fruit rows
    - Y-axis shows quantity from 0 to 80 with ticks every 10 units
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # Define people and fruit names
    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    
    # Create x positions for bars
    x = np.arange(len(people))
    width = 0.5
    
    # Plot stacked bars - bottom to top
    bottom = np.zeros(3)
    for i in range(len(fruit_names)):
        plt.bar(x, fruit[i], width, bottom=bottom, 
                color=colors[i], label=fruit_names[i])
        bottom += fruit[i]
    
    # Set labels and title
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(x, people)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.legend()
    
    plt.show()
