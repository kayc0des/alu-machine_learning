'''
Write a function def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False): that finds the best number of clusters for a GMM using the Bayesian Information Criterion:

X is a numpy.ndarray of shape (n, d) containing the data set
kmin is a positive integer containing the minimum number of clusters to check for (inclusive)
kmax is a positive integer containing the maximum number of clusters to check for (inclusive)
If kmax is None, kmax should be set to the maximum number of clusters possible
iterations is a positive integer containing the maximum number of iterations for the EM algorithm
tol is a non-negative float containing the tolerance for the EM algorithm
verbose is a boolean that determines if the EM algorithm should print information to the standard output
You should use expectation_maximization = __import__('8-EM').expectation_maximization
You may use at most 1 loop
Returns: best_k, best_result, l, b, or None, None, None, None on failure
best_k is the best value for k based on its BIC
best_result is tuple containing pi, m, S
pi is a numpy.ndarray of shape (k,) containing the cluster priors for the best number of clusters
m is a numpy.ndarray of shape (k, d) containing the centroid means for the best number of clusters
S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for the best number of clusters
l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log likelihood for each cluster size tested
b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value for each cluster size tested
Use: BIC = p * ln(n) - 2 * l
p is the number of parameters required for the model : number-of-parameters-to-be-learned-in-k-guassian-mixture-model
n is the number of data points used to create the model
l is the log likelihood of the model
'''

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    if kmax is None:
        kmax = len(X)
    if kmin < 1 or kmax < kmin or not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None, None, None
    l = []
    b = []
    for k in range(kmin, kmax + 1):
        pi, m, S = expectation_maximization(X, k, iterations, tol, verbose)
        p = k * d * (d + 1) / 2 + k - 1
        n = len(X)
        l.append(log_likelihood(X, pi, m, S))
        b.append(p * np.log(n) - 2 * l[-1])
    best_k = np.argmin(b) + kmin
    best_result = (pi, m, S)
    return best_k, best_result, np.array(l), np.array(b)