#!/usr/bin/env python3
""" Neural Network """


import numpy as np


class NeuralNetwork(object):
    ''' Neural Network '''

    def __init__(self, nx, nodes):
        '''
        Constructor method
        Param: nx -> number of inputs
        nodes -> number of neurons in hidden layer
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, self.nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getter function to access the private attributes
    @property
    def W1(self):
        ''' Return W1 '''
        return self.__W1

    @property
    def b1(self):
        ''' Return b1 '''
        return self.__b1

    @property
    def A1(self):
        ''' Return A1 '''
        return self.__A1

    @property
    def W2(self):
        ''' Return W2 '''
        return self.__W2

    @property
    def b2(self):
        ''' Return b2 '''
        return self.__b2

    @property
    def A2(self):
        ''' Return A2 '''
        return self.__A2

    def forward_prop(self, X):
        '''
        Peforms forward propagation
        '''
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        ''' Calculates the cost of the model'''
        m = Y.shape[1]  # number of examples
        sum = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = -1/m * sum
        return cost

    def evaluate(self, X, Y):
        ''' Evaluates the networks prediction '''
        A1, A2 = self.forward_prop(X)
        prediction = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return prediction, cost

lib_train = np.load('data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NeuralNetwork(X.shape[0], 3)
A, cost = nn.evaluate(X, Y)
print(A)
print(cost)