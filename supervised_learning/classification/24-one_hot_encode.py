#!/usr/bin/env python3
'''
converts a numeric label vector into a one-hot matrix
'''


import numpy as np


def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot matrix.

    Arguments:
    Y: numpy.ndarray, shape (m,), numeric class labels
    classes: int, maximum number of classes found in Y

    Returns:
    one-hot encoding of Y with shape (classes, m), or None on failure
    """

    # Check if the input is valid
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None

    m = Y.shape[0]  # number of examples
    one_hot_matrix = np.zeros((classes, m))

    # Set the appropriate element to 1 for each class label
    for i in range(m):
        if Y[i] >= classes or Y[i] < 0:
            return None  # Invalid label
        one_hot_matrix[Y[i], i] = 1

    return one_hot_matrix
