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

    gradients = {}

    for i in reversed(range(1, L + 1)):
        A_current = cache['A' + str(i)]
        A_previous = cache['A' + str(i - 1)] if i > 1 else cache['A0']

        if i == L:
            dZ = A_current - Y
        else:
            dA = np.dot(weights
                           ['W' + str(i + 1)].T, gradients['dZ' + str(i + 1)])
            dZ = dA * (1 - (A_current * A_current))

        gradients['dZ' + str(i)] = dZ
        gradients['dW' + str(i)] = (1 / m) * np.dot(
            dZ, A_previous.T) + (lambtha / m) * weights['W' + str(i)]
        gradients['db' + str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights['W' + str(i)] -= alpha * gradients['dW' + str(i)]
        weights['b' + str(i)] -= alpha * gradients['db' + str(i)]

    return weights
