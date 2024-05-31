#!/usr/bin/env python3
''' Gradient descent with L2 regularization '''


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''
    Updates weights and biases using
    Gradient descent with L2 regularization

    Params:
    Y -> one hot array of shape (classes, m)
    weights -> dict of weights and biases
    cache -> output of each layer
    alpha -> learning rate
    lambtha -> l2 reg parameter
    L -> number of layers in the network
    activation -> output: softmax, hidden: tanh

    Returns:
    Updated weights and biases
    '''

    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dw = (1 / m) * np.matmul(dz, A.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(W.T, dz) * (1 - np.square(A))
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dw
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db

    return weights
