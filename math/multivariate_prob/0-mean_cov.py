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
  """
  Calculates the mean and covariance of a data set.

  Args:
      X: A 2D numpy.ndarray of shape (n, d) containing the data set.

  Returns:
      A tuple containing two numpy.ndarrays:
          mean: A 1D array of shape (1, d) containing the mean of the data set.
          cov: A 2D array of shape (d, d) containing the covariance matrix.

  Raises:
      TypeError: If X is not a 2D numpy.ndarray.
      ValueError: If X contains less than 2 data points.
  """

  # Check if X is a 2D numpy.ndarray
  if not isinstance(X, np.ndarray) or X.ndim != 2:
    raise TypeError("X must be a 2D numpy.ndarray")

  # Check if X has multiple data points
  n, d = X.shape
  if n < 2:
    raise ValueError("X must contain multiple data points")

  # Calculate the mean
  mean = np.mean(X, axis=0)  # Mean along columns
  mean = np.array(mean)

  # Calculate the centered data matrix
  centered_X = X - mean

  # Calculate the covariance matrix using manual computation
  cov = (1 / (n - 1)) * np.dot(centered_X.T, centered_X)
  cov = np.array(cov)

  return mean, cov

print(mean_cov(X))