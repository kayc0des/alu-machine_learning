#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Define fruit names and colors
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Create a stacked bar graph
fig, ax = plt.subplots()
print(fruit)

# Plot each person's fruit quantities
bottom_values = np.zeros(3)  # Initialize the bottom values array
for i in range(4):
    print(fruit[i, :])
    ax.bar(range(3), fruit[i, :], bottom=bottom_values, label=f'{fruit_names[i]}', color=colors[i], width=0.5)
    bottom_values += fruit[i, :]  # Update bottom values for the next iteration


ax.set_xlabel('Person')
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')

# Set y-axis range and ticks
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))

# Add legend
ax.legend()

# Set x-axis ticks and labels
ax.set_xticks(range(3))
ax.set_xticklabels(['Farrah', 'Fred', 'Felicia'])

# Show the plot
plt.show()
