#!/usr/bin/env python3
'''
converts a numeric label vector into a one-hot matrix
'''


import numpy as np


import numpy as np


def one_hot_decode(one_hot):
    """
    Convert a one-hot matrix into a vector of labels.

    Arguments:
    one_hot: numpy.ndarray, one-hot encoded matrix with shape (classes, m)

    Returns:
    numpy.ndarray with shape (m, ) 
    containing the numeric labels for each example,
    or None on failure
    """

    # Check if the input is valid
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    classes, m = one_hot.shape

    # Initialize an empty array to store the decoded labels
    decoded_labels = np.zeros(m)

    # Decode the one-hot matrix
    for i in range(m):
        label_index = np.argmax(one_hot[:, i])
        decoded_labels[i] = label_index

    return decoded_labels.astype(int)
