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


if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(m)
    print(s)