#!/usr/bin/env python3
''' Forward propagation for a simple RNN '''


import numpy as np


def rnn(rnn_cell, X, h_0):
    '''
    This function performs forward
    propagation for a simple RNN

    Args:
        X - data to be used
        h_0 - initial hidden state

    Returns:
        H - np array conatining all of the hidden states
        Y - np array containing all of the outputs
    '''

    t, m, i = X.shape
    m, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        if step == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))

    output_shape = Y.shape[-1]
    Y = Y.reshape(t, m, output_shape)

    return (H, Y)
