#!/usr/bin/env python3
''' Evaluates l2 loss of a model '''


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''
    Calculates the cost of a neural network with
    L2 regularization

    Param:
    cost: cost without L2 regularization
    lambtha: regularization parameter
    weights: dict of the weights and biases
    L: number of layers in the neural network
    m: number of data points used

    Returns:
    Cost of the network accounting for L2 regularization
    '''

    frobenius_norm = 0
    for i in range(L):
        val = np.dot(weights['W' + str(i)], weights['W' + str(i)])
        val = np.sum(val)
        frobenius_norm += val

    penalty_term = (lambtha / (2 * m)) * frobenius_norm
    l2_loss = cost + penalty_term

    return l2_loss

