#!/usr/bin/env python3
"""
This scripts use the scatter method to plot values
"""


import numpy as np
import matplotlib.pyplot as plt


mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# define x,y and title values - use scatter to plot
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.title("Men's Height vs Weight")

plt.scatter(x, y, color='magenta', s=10)
plt.show()
