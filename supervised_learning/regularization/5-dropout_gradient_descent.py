#!/usr/bin/env python3
'''
Gradient descent with drop out
'''


import numpy as np


def dropout_gradient_descent(
    Y, weights, cache, alpha, keep_prob, L):
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
    
    m = Y.sahpe[1]
    
    for i in reversed(range(1, L + 1)):
        # do some
        print('do some')

