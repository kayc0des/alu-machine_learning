#!/usr/bin/env python3
'''
Expectation step in the EM algorithm for a GMM
'''


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    '''
    Expectation step in the EM algorithm for a GMM
    
    Args:
        X (np.ndarray): Data set
        pi (np.ndarray): Priors for each cluster
        m (np.ndarray): Centroid means for each cluster
        S (np.ndarray): Covariance matrices for each cluster
    Returns:
        g (np.ndarray): Posterior probabilities for each data point in each cluster
        l (float): Total log likelihood
    '''
    if not isinstance(X, np.ndarray):
        return None, None
    if not isinstance(pi, np.ndarray):
        return None, None
    if not isinstance(m, np.ndarray):
        return None, None
    if not isinstance(S, np.ndarray):
        return None, None

    k, d = m.shape
    n = X.shape[0]
    g = np.zeros((k, n))
    l = 0
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    l = np.sum(np.log(np.sum(g, axis=0)))
    g /= np.sum(g, axis=0)
    return g, l
