#!/usr/bin/env python3
"""
This script defines a class Neuron that defines
a single neuron performing binary classification.
"""


import numpy as np


class Neuron(object):
    """ Define's a single Neuron """

    def __init__(self, nx):
        """
        Neuron Class Constructor
        Param: nx -> is the number of input features.
        """

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx

        # Set private instance attributes
        self.__W = np.random.randn(1, self.nx)
        self.__b = 0
        self.__A = 0

    # Getter function to access the private attributes
    @property
    def W(self):
        ''' Return W '''
        return self.__W

    @property
    def b(self):
        ''' Return b '''
        return self.__b

    @property
    def A(self):
        ''' Return A '''
        return self.__A

    def forward_prop(self, X):
        ''' Method performs forward propagation '''
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        ''' Calculates the cost of the model'''
        m = Y.shape[1]  # number of examples
        sum = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = -1/m * sum
        return cost

    def evaluate(self, X, Y):
        ''' Evaluates the neuron's prediction '''
        A_hat = self.forward_prop(X)
        A = np.where(A_hat > 0.5, 1, 0)
        cost = self.cost(Y, A_hat)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.5):
        ''' Calculates one pass of gradient descent '''
        m = Y.shape[1]
        # Evaluate the partial derivatives of the cost function
        dz = A - Y
        dw = (1 / m) * np.matmul(X, dz.T)
        db = (1 / m) * np.sum(dz)

        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)

        return self.__W, self.__b
