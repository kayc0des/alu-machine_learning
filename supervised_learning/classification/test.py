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
        m = self.nx # number of examples
        cost = -1/m * np.sum((Y * np.log(A)) + ((1.000001 - Y) * np.log(1.0000001 - A)))
        return cost
    
    def evaluate(self, X, Y):
        ''' Evaluates the neuron's prediction '''
        A_hat = self.forward_prop(X)
        A = np.where(A_hat > 0.5, 1, 0)
        cost = self.cost(Y, A_hat)
        return A, cost
    
    def gradient_descent(self, X, Y, A, alpha=0.5):
        ''' Calculates one pass of gradient descent '''
        # Evaluate the partial derivatives of the cost function
        A = A.reshape(-1, 1)
        dz = A - Y
        print(f'shape of A -> {A.T.shape}')
        print(f'shape of Y -> {Y.shape}')
        dw = np.dot(X, dz.T)
        db = dz
        # update the private attributes __W and __b
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
        print(f'shape __W -> {self.__W.shape}')
        return self.__W, self.__b


''' Debug '''
lib_train = np.load('data/Binary_Train.npz') # load dataset
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

# ------------------------------------------ #
print(f'The shape of X_3D is -> {X_3D.shape}')
print(f'The shape of Y is -> {Y.shape}')
print(f'The shape of X is -> {X.shape}')
# ------------------------------------------ #

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
neuron.gradient_descent(X, Y, A, 0.5)
print(neuron.W)
print(neuron.b)


# test = np.array([[[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9],
#                  [1, 3, 4]], 
#                  [[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9],
#                  [1, 3, 4]],])

# new_array = test.reshape((test.shape[0], -1)).T
# print(test.shape)
# print(new_array)
# print(new_array.shape)
