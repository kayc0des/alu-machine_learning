#!/usr/bin/env python3
"""
This script plots x -> y1 and x -> y2
"""


import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.plot(x, y1, color='red', linestyle='--', label='C-14')
plt.plot(x, y2, color='green', linestyle='-', label='Ra-226')
plt.legend()

# define x,y and title values - use scatter to plots
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title("Exponential Decay of C-14")

# The x-axis should range from 0 to 28650
plt.xlim(0, 20000)
plt.ylim(0, 1)

plt.show()
