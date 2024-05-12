#!/usr/bin/env python3
"""
This script defines a class Neuron that defines
a single neuron performing binary classification.
"""


import numpy as np


class Neuron():
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
        self._W = np.random.randn(1, self.nx)
        self._b = 0
        self._A = 0

    # Getter function to access the private attributes
    def get_W(self):
        ''' Return W '''
        return self._W

    def get_b(self):
        ''' Return b '''
        return self._b

    def get_A(self):
        ''' Return A '''
        return self._A
