#!/usr/bin/env python3
"""
This script using matplotlib.pyplot to plot values for x and y
"""


import numpy as np
import matplotlib.pyplot as plt


y = np.arange(0, 11) ** 3 
x = np.arange(0, 11)

# plot x and y, and give the color param a value of red
plt.plot(x, y, color='red', linestyle='-')
plt.show()
