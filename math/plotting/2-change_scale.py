#!/usr/bin/env python3
"""
This script changes the scale
"""


import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y)

# define x,y and title values - use scatter to plot
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title("Exponential Decay of C-14")

# The x-axis should range from 0 to 28650
plt.xlim(0, 28650)

# Y-axis should be logarithmically scaled
plt.yscale('log')

plt.show()
