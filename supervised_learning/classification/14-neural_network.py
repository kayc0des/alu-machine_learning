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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        ''' One pass of gradient descent '''
        m = Y.shape[1]

        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)

        # Gradient descent -> output layer
        # dz2 = A2 - Y
        # dw2 = (1 / m) * np.dot(A1, dz2.T)
        # db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        # # Update weights and bias of the output layer
        # self.__W2 = self.__W2 - (alpha * dw2.T)
        # self.__b2 = self.__b2 - (alpha * db2)

        # # Backpropagate the error to compute gradients -> hidden layer
        # dz1 = np.dot(self.__W2.T, dz2) * (A1 * (1 - A1))
        # dw1 = (1 / m) * np.dot(X, dz1.T)
        # db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        # # Update weights and bias of the hidden layer
        # self.__W1 = self.__W1 - (alpha * dw1.T)
        # self.__b1 = self.__b1 - (alpha * db1)

        return self.__W1, self.__b1, self.__W2, self.__b2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        ''' Trains the network '''

        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            A = self.forward_prop(X)

            # gradient descent
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        evaluation = self.evaluate(X, Y)
        return evaluation
