#!/usr/bin/env python3
''' Script performs PCA on a dataset '''


import numpy as np


def pca(X, var=0.95):
    '''
    Performs PCA on a dataset

    Args:
        X -> np array of shape (n, d)
        var -> fraction of the variance that
            the PCA transformation should maintain

    Returns:
        W -> Weights matrix of shape (d, nd) that maintains
            var function of X's ooriginal variance
        nd -> the new dimensionality of the transformed X
    '''
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_idx]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]

    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)

    nd = np.searchsorted(cumulative_variance, var) + 1
    
    W = sorted_eigenvectors[:, :nd]

    return W
