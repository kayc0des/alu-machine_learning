#!/usr/bin/env python3
'''
Conducts forward prop using dropout
'''


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''
    Conducts forward prop + dropout

    Param:
    X -> numpy array of shape (nx, m)
    weights -> dict of weights
    L -> number of layers in the network
    keep_prob -> probability that a node will be kept

    Returns:
    Dict with outputs of each layer and dropout mask
    '''

    cache = {}
    cache['A0'] = X

    # evaluate forward prop per layer
    for i in range(1, L + 1):
        if i == 1:
            z = np.matmul(
                weights['W' + str(i)], X) + weights['b' + str(i)]
        else:
            layer_input = cache['A' + str(i-1)]
            z = np.matmul(
                weights['W' + str(i)],
                layer_input) + weights['b' + str(i)]

        if i == L:
            A = np.exp(z) / np.sum(np.exp(z), axis=0)
            cache['A' + str(i)] = A
        else:
            A = np.tanh(z)
            # apply dropout mask to hidden layers
            dropout_mask = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A *= dropout_mask
            # Scale the remaining neurons to maintain expected value
            A /= keep_prob
            cache['A' + str(i)] = A
            cache['D' + str(i)] = dropout_mask

    return cache
