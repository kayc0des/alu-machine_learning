#!/usr/bin/env python3


import numpy as np


def pdf(X, m, S):
    """
    Calculate the probability density function (PDF) of a Gaussian distribution.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        The data points whose PDF should be evaluated.
    m : numpy.ndarray of shape (d,)
        The mean of the distribution.
    S : numpy.ndarray of shape (d, d)
        The covariance matrix of the distribution.

    Returns:
    P : numpy.ndarray of shape (n,)
        The PDF values for each data point, or None on failure.
    """
    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray):
        return None
    if len(X.shape) != 2 or len(m.shape) != 1 or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1] or S.shape[0] != m.shape[0]:
        return None

    # Get dimensions
    n, d = X.shape

    # Compute the determinant and inverse of the covariance matrix
    det_S = np.linalg.det(S)
    if det_S == 0:
        return None
    inv_S = np.linalg.inv(S)

    # Compute the constant factor in the PDF equation
    denom = np.sqrt((2 * np.pi) ** d * det_S)

    # Center the data by subtracting the mean
    X_centered = X - m

    # Compute the exponent (this uses matrix multiplication)
    exponent = np.sum(X_centered @ inv_S * X_centered, axis=1)

    # Compute the PDF values
    P = (1. / denom) * np.exp(-0.5 * exponent)

    # Ensure minimum value of 1e-300 for numerical stability
    P = np.maximum(P, 1e-300)

    return P
