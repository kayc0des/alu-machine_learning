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

    try:
        n, d = X.shape

        # Compute the determinant and inverse of the covariance matrix
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)

        # Calculate the constant coefficient in the Gaussian PDF formula
        denom = np.sqrt(((2 * np.pi) ** d) * det_S)

        # Compute the exponent part
        X_m = X - m
        exponent = -0.5 * np.einsum('ij,ij->i', X_m @ inv_S, X_m)

        # Calculate the PDF for each data point
        P = (1. / denom) * np.exp(exponent)

        # Ensure a minimum value of 1e-300 for all elements in P
        P = np.maximum(P, 1e-300)

        return P
    except Exception as e:
        return None
