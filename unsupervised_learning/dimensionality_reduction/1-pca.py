#!/usr/bin/env python3
''' Script performs PCA on a dataset '''


import numpy as np


def pca(X, ndim):
    '''
    Performs PCA on a dataset

    Args:
        X -> np array of shape (n, d)
        ndim -> new dimensionality of X

    Returns:
        T -> np array of shape (d, ndim)
        containing transformed version of X
    '''
    cov_matrix = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]

    W = sorted_eigenvectors[:, :ndim]

    T = np.dot(X, W)

    return T
