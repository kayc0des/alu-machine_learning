#!/usr/bin/env python3
""" This script defines a class Neuron that defines
a single neuron performing binary classification """


import numpy as np

class Neuron():
    """ Define's a single Neuron """
    
    # declare public instance attributes
    W = np.random.randn(0, 1)
    b = 0
    A = 0
    
    def __init__(self, nx):
        """ Class Constructor
        Param: nx -> is the number of input features """
        
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
