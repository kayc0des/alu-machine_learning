#!/usr/bin/env python3
''' Shuffles data points in two matrices '''


import numpy as np


def shuffle_data(X, Y):
    '''
    Shuffles data points in two matrices the same way

    Args:
    X -> np.ndarray of shape (m, nx)
    Y -> np.ndarray of shape (m, ny)

    Returns:
    The shuffled X and Y matrices
    '''
    m = X.shape[0]
    permuted_indices = np.random.permutation(m)

    #shuffle both arrays
    shuffled_X = X[permuted_indices]
    shuffled_Y = Y[permuted_indices]

    return shuffled_X, shuffled_Y
    