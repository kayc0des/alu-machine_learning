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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(1, self.__L + 1):
            if i == 1:
                self.__weights['W' + str(i)] = np.random.randn(
                    layers[i - 1], nx) * np.sqrt(2/nx)
            else:
                self.__weights['W' + str(i)] = np.random.randn(
                    layers[i - 1], layers[i - 2]) * np.sqrt(2/layers[i - 2])
            self.__weights['b' + str(i)] = np.zeros((layers[i - 1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        ''' Performs forward propagation '''

        # loop through weights{} to compute A
        for i in range(1, self.__L + 1):
            if i == 1:
                z = np.dot(self.__weights[f'W{i}'], X) + self.__weights[f'b{i}']
                a = 1 / (1 + np.exp(-z))
            else:
                z = np.dot(self.__weights[f'W{i}'], self.__cache[f'A{i - 1}']) + self.__weights[f'b{i}']
                a = 1 / (1 + np.exp(-z))
            self.__cache[f'A{i}'] = a

        # A0 -> X input
        self.__cache['A0'] = X
        # final output A -> last evaluation of forward prop
        A = self.__cache[f'A{self.__L}']

        return A, self.__cache
    
lib_train = np.load('data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = DeepNeuralNetwork(X.shape[0], [5, 3, 1])
deep._DeepNeuralNetwork__weights['b1'] = np.ones((5, 1))
deep._DeepNeuralNetwork__weights['b2'] = np.ones((3, 1))
deep._DeepNeuralNetwork__weights['b3'] = np.ones((1, 1))
A, cache = deep.forward_prop(X)
print(A)
print(cache)
print(cache is deep.cache)
print(A is cache['A3'])