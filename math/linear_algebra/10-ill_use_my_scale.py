#!/usr/bin/env python3
"""
This script provides a function to evaluate the shape of a matrix
"""


import numpy as np


def np_shape(matrix):
    """
    This function returns the shape of a matrix

    Paramters:
    - matrix: an ndarray

    Returns:
    - tuple: a tuple representing the shape of a matrix
    """
    return np.shape(matrix)
