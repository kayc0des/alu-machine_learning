#!/usr/bin/env python3
"""
all in one
"""


import numpy as np
import matplotlib.pyplot as plt


y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create a 3x2 subplot grid
fig, axes = plt.subplots(3, 2)

# Line
axes[0, 0].plot(np.arange(0, 11), y0, color='red')

# Scatter
axes[0, 1].scatter(x1, y1, color='magenta', s=20)
axes[0, 1].set_xlabel('Height (in)', fontsize='x-small')
axes[0, 1].set_ylabel('Weight (lbs)', fontsize='x-small')
axes[0, 1].set_title("Men's Height vs Weight", fontsize='x-small')

# Change the scale
axes[1, 0].plot(x2, y2)
axes[1, 0].set_xlabel('Time (years)', fontsize='x-small')
axes[1, 0].set_ylabel('Fraction Remaining', fontsize='x-small')
axes[1, 0].set_yscale('log')
axes[1, 0].set_title('Exponential Decay of C-14', fontsize='x-small')

# Exponential decay
axes[1, 1].plot(x3, y31, label='C-14', linestyle='--', color='red')
axes[1, 1].plot(x3, y32, label='Ra-226', linestyle='-', color='green')
axes[1, 1].set_xlabel('Time (years)', fontsize='x-small')
axes[1, 1].set_ylabel('Fraction Remaining', fontsize='x-small')
axes[1, 1].legend()
axes[1, 1].set_title('Exponential Decay of Radioactice Elements', fontsize='x-small')

# Histogram
axes[2, 0].hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
axes[2, 0].set_ylim(0, 31)
axes[2, 0].set_xlabel('Grades', fontsize='x-small')
axes[2, 0].set_ylabel('Number of Students', fontsize='x-small')
axes[2, 0].set_title('Project A', fontsize='x-small')

# Remove empty subplot in the last row and last column
fig.delaxes(axes[2, 1])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
