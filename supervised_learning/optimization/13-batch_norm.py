#!/usr/bin/env python3
''' Normalizes an unactivated output '''


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''
    Normalizes an unactivated output of a neural
    network using batch normalization

    Args:
    Z: numpy.ndarray of shape (m, n) that should be normalized
        m -> number of data points
        n -> the number of features in Z
    gamma: numpy.ndarray of shape (1, n) containing
        the scales used for batch normalization
    beta: numpy.ndarray of shape (1, n) containing
        the offsets used for batch normalization
    epsilon: small number used to avoid division by zero

    Returns:
    Normalized matrix z
    '''

    # Evaluate the mean and the variance
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # Normalize Z
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    
    # Apply a different distribution to Z_norm
    Z_tilde = gamma * Z_norm + beta

    return Z_tilde
