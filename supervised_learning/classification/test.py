#!/usr/bin/env python3
"""
This script defines a class Neuron that defines
a single neuron performing binary classification.
"""


import numpy as np
import matplotlib.pyplot as plt



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
        '''
        Method performs forward propagation
        '''
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        '''
        Calculates the cost of the model
        '''
        m = Y.shape[1]  # number of examples
        sum = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = -1/m * sum
        return cost

    def evaluate(self, X, Y):
        '''
        Evaluates the neuron's prediction
        '''
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''
        Calculates one pass of gradient descent
        '''
        m = Y.shape[1]

        # Evaluate the partial derivatives of the cost function
        dz = A - Y
        dw = (1 / m) * np.dot(X, dz.T)
        db = (1 / m) * np.sum(dz)

        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)

        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''
        Trains the Neuron
        '''
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            # Forward propagation
            A = self.forward_prop(X)

            # gradient descent
            self.gradient_descent(X, Y, A, alpha)

        # Evaluation of training
        evaluation = self.evaluate(X, Y)

        return self.__W, self.__b, evaluation

''' Debug '''
# lib_train = np.load('data/Binary_Train.npz') # load dataset
# X_3D, Y = lib_train['X'], lib_train['Y']
# X = X_3D.reshape((X_3D.shape[0], -1)).T

# print(Y.shape)

# np.random.seed(0)
# neuron = Neuron(X.shape[0])
# A = neuron.forward_prop(X)
# print(A.shape)
# # neuron.gradient_descent(X, Y, A, 0.5)
# # print(neuron.W)
# # print(neuron.b)

lib_train = np.load('data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=10)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", np.round(cost, decimals=10))
print("Train accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Train data:", np.round(A, decimals=10))
print("Train Neuron A:", np.round(neuron.A, decimals=10))

A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", np.round(cost, decimals=10))
print("Dev accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Dev data:", np.round(A, decimals=10))
print("Dev Neuron A:", np.round(neuron.A, decimals=10))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()



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
