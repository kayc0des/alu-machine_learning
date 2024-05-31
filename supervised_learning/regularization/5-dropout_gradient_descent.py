#!/usr/bin/env python3
'''
Gradient descent with drop out
'''


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''
    Updates weights of a neural network with dropout

    Param:
    Y -> one how array with shape (classes, m)
    weights -> dict of the weights and biases
    cache -> output and dropdout masks
    alpha -> learning rate
    keep_prop -> propbability a node will be kept
    L -> number of layers

    Returns:
    Weights of the network updated
    '''
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA_prev = np.dot(W.T, dZ)
            D = cache['D' + str(i - 1)]
            dA_prev *= D  # Apply dropout mask
            dA_prev /= keep_prob  # Scale the activation back up
            dZ = dA_prev * (1 - np.power(A_prev, 2))  # Derivative of tanh

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

    return weights
