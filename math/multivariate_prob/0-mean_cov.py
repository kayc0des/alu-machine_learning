#!/usr/bin/env python3
'''
mean_cov(X) calculates the mean and variance
of a data set'''


import numpy as np


np.random.seed(0)
X = np.random.multivariate_normal([12,30,10], [[36, -30, 15],
                                               [-30,100,-20],
                                               [15, -20, 25]], 10000)

def mean_cov(X):
    '''
    Returns the mean and covariance of a data set
    '''
    # Check if X is a 2D numpy.ndarray
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    
    # Get the number of data points (n) and the number of dimensions (d)
    n, d = X.shape
    
    # Check if there are multiple data points
    if n < 2:
        raise ValueError("X must contain multiple data points")
    
    # Compute the mean of the data set
    mean = np.mean(X, axis=0, keepdims=True)
    
    # Compute the covariance matrix of the data set
    cov = np.dot((X - mean).T, X - mean) / (n - 1)
    
    return mean, cov

print(mean_cov(X))