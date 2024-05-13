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
        m = Y.shape[0]
        cost = 1/m * (-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))
        return cost
