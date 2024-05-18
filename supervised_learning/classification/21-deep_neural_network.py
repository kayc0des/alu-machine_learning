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
        ''' Return L '''
        return self.__L

    @property
    def cache(self):
        ''' Return cache '''
        return self.__cache

    @property
    def weights(self):
        ''' Return weights'''
        return self.__weights

    def forward_prop(self, X):
        ''' Performs forward propagation '''

        # A0 -> X input
        self.__cache['A0'] = X

        # loop through weights{} to compute A
        for i in range(1, self.__L + 1):
            if i == 1:
                z = np.dot(self.__weights['W{}'.format(i)],
                           X) + self.__weights['b{}'.format(i)]
                a = 1 / (1 + np.exp(-z))
            else:
                z = np.dot(self.__weights['W{}'.format(i)],
                           self.__cache['A{}'.format(
                               i - 1)]) + self.__weights['b{}'.format(i)]
                a = 1 / (1 + np.exp(-z))
            self.__cache['A{}'.format(i)] = a

        # final output A -> last evaluation of forward prop
        A = self.__cache['A{}'.format(self.__L)]

        return A, self.__cache

    def cost(self, Y, A):
        ''' Calculates the models cost '''
        m = Y.shape[1]  # number of examples
        sum = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = -1/m * sum
        return cost

    def evaluate(self, X, Y):
        ''' Evaluate the neural network's prediction '''
        A, cache = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        ''' Calculates one pass of gradient descent '''
        m = Y.shape[1]
        L = self.__L  # number of layers
        A_L = cache["A{}".format(L)]
        dZ = A_L - Y

        # Compute gradients using backpropagation
        for i in range(L, 0, -1):
            A = cache["A{}".format(i)]
            A_prev = cache["A{}".format(i - 1)]
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            # Calculate dZ for the next layer if not the input layer
            if i > 1:
                dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)
            
            # Update weights and biases
            self.__weights["W{}".format(i)] -= alpha * dW
            self.__weights["b{}".format(i)] -= alpha * db
        
        return self.__weights
