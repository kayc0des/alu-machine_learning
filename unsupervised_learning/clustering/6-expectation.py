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
        X (np.ndarray): Data set of shape (n, d)
        pi (np.ndarray): Priors for each cluster of shape (k,)
        m (np.ndarray): Centroid means for each cluster of shape (k, d)
        S (np.ndarray): Covariance matrices for each cluster of shape (k, d, d)
    Returns:
        g (np.ndarray): Posterior probabilities for each data point in each cluster
        l (float): Total log likelihood
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    
    n, d = X.shape
    k = pi.shape[0]
    
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    try:
        g = np.zeros((k, n))
        for i in range(k):
            pdf_result = pdf(X, m[i], S[i])
            if pdf_result is None:
                return None, None
            g[i] = pi[i] * pdf_result
        
        total_likelihood = np.sum(g, axis=0)
        if np.any(total_likelihood == 0):
            return None, None
        
        l = np.sum(np.log(total_likelihood))
        g /= total_likelihood
        return g, l
    except:
        return None, None
