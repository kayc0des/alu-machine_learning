#!/usr/bin/env python3
''' RNNCell represents a cell of a simple RNN '''


import numpy as np


class RNNCell():
    ''' This class represents a cell of a simple RNN '''

    def __init__(self, i, h, o):
        '''
        This method creates an instance of the RNNCell

        Args:
            i - the dimensionality of the data
            h - the dimensionality of the hidden state
            o - the dimensionality of the outputs

        Returns:
            h_next - the next hidden state
            y - the output of the cell
        '''

        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Performs the softmax function

        parameters:
            x: the value to perform softmax on to generate output of cell

        return:
            softmax of x
        """

        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)

        return softmax

    def forward(self, h_prev, x_t):
        '''
        Function that performs forward propagation

        Args:
            h_prev - previous hidden state
            x_t - input data at time step t

        Returns:
            h_next - next hidden state
            y - the output of the sell
        '''

        concatenation = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenation, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
