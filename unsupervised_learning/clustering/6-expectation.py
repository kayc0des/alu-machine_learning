#!/usr/bin/env python3
'''
Expectation step in the EM algorithm for a GMM
'''

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    if not isinstance(X, np.ndarray) or not isinstance(pi, np.ndarray) or not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray):
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
