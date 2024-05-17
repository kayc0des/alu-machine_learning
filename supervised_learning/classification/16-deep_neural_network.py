#!/usr/bin/env python3
'''
Deep Neural Network for Binary Classification
'''


import numpy as np


class DeepNeuralNetwork(object):
    '''
    Define's a Deep Neural Network Class
    '''

    def __init__(self, nx, layers):
        '''
        Class constructor
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            if l == 1:
                self.weights['W' + str(l)] = np.random.randn(layers[l - 1], nx) * np.sqrt(2/nx)
            else:
                self.weights['W' + str(l)] = np.random.randn(layers[l - 1], layers[l - 2]) * np.sqrt(2/layers[l - 2])
            self.weights['b' + str(l)] = np.zeros((layers[l - 1], 1))
