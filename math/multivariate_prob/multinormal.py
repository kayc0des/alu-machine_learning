#!/usr/bin/env python3
"""
This script represents a Multivariate normal distributiion
"""


# import modules
import numpy as np


class MultiNormal():
    """
    This class reps a Multivariate Gaussian Distribution
    """

    def __init__(self, data):
        """ Init method """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        n, d = data.shape

        # ensure data contains multiple data points
        if n < 2:
            raise ValueError('data must contain multiple data points')

        # set public instance variables mean and cov
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.cov = np.dot((data - self.mean).T, data - self.mean) / (n - 1)
 