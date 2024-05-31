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

    grad_cache = {}

    for i in reversed(range(1, L+1)):
        current_a = cache['A' + str(i)]
        prev_a = cache['A' + str(i - 1)]

        if i == L:
            # gradient of loss w.r.t output layer
            grad_cache['dZ' + str(i)] = current_a - Y
        else:
            grad_cache['dA' + str(i)] = np.matmul(weights['W' + str(i 
                                                                    + 1)].T, grad_cache['dZ' + str(i + 1)])
            val = 1 - np.square(current_a)
            grad_cache['dZ' + str(i)] = grad_cache['dA' + str(i)] * val

        layer_output_grad = grad_cache['dZ' + str(i)]
        dW = (np.matmul(layer_output_grad, prev_a.T) / m) + ((lambtha
                                                              / m) * weights['W' + str(i)])
        dB = np.sum(layer_output_grad, axis=1, keepdims=True) / m

        # update weights and biases
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * dB

    return weights
