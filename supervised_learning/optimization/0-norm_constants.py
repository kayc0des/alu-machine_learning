#!/usr/bin/env python3
''' Evaluate standardization constants '''


import numpy as np


def normalization_constants(X):
    '''
    Calculates the normalization constants
    of a matrix

    Args:
    X -> a numpy.ndarray with shape (m, nx)
    m -> number of examples
    nx -> number of features

    Returns:
    mean and standard deviation'''

    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)

    return mean, stddev
