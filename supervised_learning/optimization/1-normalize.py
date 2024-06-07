#!/usr/bin/env python3
''' Normalizes a matrix '''


import numpy as np


def normalize(X, m, s):
    '''
    Normalizes a matrix

    Args:
    X -> np.ndarray of shape (d, nx)
    m -> np.ndarray containing mean values
    s -> np.ndarray containing std values

    Returns:
    Normalized matrix
    '''
    return (X - m ) / s
