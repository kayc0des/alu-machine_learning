#!/usr/bin/env python3
"""
Histogram Plot
"""


import numpy as np
import matplotlib.pyplot as plt


np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')

plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title("Project A")

plt.xlim(0, 100)
plt.show()
