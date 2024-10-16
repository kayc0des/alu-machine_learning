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
        self.W1 = np.random.randn(nodes, self.nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
