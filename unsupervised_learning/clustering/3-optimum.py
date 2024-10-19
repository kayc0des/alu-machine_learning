#!/usr/bin/env python3

"""
This module contains a function that
tests for the optimum number of clusters by variance
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''
    Test for the optimum number of clusters by variance
    
    Args:
        X: numpy.ndarray of shape (n, d)
        containing the data set
        kmin: positive integer
        containing the minimum number of clusters to check for (inclusive)
        kmax: positive integer
        containing the maximum number of clusters to check for (inclusive)
        iterations: positive integer
        containing the maximum number of iterations for K-means
    Returns:
        results: list containing the
        outputs of K-means for each cluster size
        d_vars: list containing the difference in
        variance from the smallest cluster size for each cluster size
    '''
    if kmax is None:
        kmax = X.shape[0]
    results, d_vars = [], []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        d_vars.append(variance(X, C))
    return results, [d_vars[0] - x for x in d_vars]
