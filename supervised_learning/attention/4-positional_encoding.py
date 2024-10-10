#!/usr/bin/env python3
''' Positional Encoding '''


import numpy as np


def positional_encoding(max_seq_len, dm):
    '''
    Calculates the positional encoding for a transformer

    Args:
        max_seq_len: int representing the maximum sequence length
        dm: int representing the model depth

    Returns:
        positional_encoding: numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectors  
    '''
    positional_encoding = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(dm):
            if j % 2 == 0:
                positional_encoding[i, j] = np.sin(i / 10000 ** (j / dm))
            else:
                positional_encoding[i, j] = np.cos(i / 10000 ** ((j - 1) / dm))

    return positional_encoding
