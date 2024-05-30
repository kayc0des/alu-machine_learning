#!/usr/bin/env python3
''' Gradient descent with L2 reg '''


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

    temp_dict = {}

    for i in reversed(range(1, L+1)):
        if i == L:
            temp_dict['dZ' + str(i)] = cache['A' + str(i)] - Y
            temp_dict['dW' + str(i)] = (( 1 / m) * np.matmul(temp_dict['dZ' + str(i)], cache['A' + str(i -1)].T)) + (lambtha / m) * weights['W' + str(i)]
            temp_dict['dB' + str(i)] = (1 / m) * np.sum(temp_dict['dZ' + str(i)])
        else:
            temp_dict['dA' + str(i)] = np.matmul(weights['W' + str(i + 1)].T, temp_dict['dZ' + str(i + 1)])
            value = 1 - np.sqaured(cache['A' + str(i)]
            temp_dict['dZ' + str(i)] = temp_dict['dA' + str(i)] * value
            temp_dict['dW' + str(i)] = (( 1 / m) * np.matmul(
                temp_dict['dZ' + str(i)], cache['A' + str(i - 1)].T)) + (lambtha / m) * weights['W' + str(i)]
            temp_dict['dB' + str(i)] = (1 / m) * np.sum(temp_dict['dZ' + str(i)])

        # update weights and biases
        weights['W' + str(i)] -= alpha * temp_dict['dZ' + str(i)]
        weights['b' + str(i)] -= alpha * temp_dict['dB' + str(i)]

    return weights
