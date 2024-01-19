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
    ndim = matrix.ndim
    return (
        matrix.shape[0] * (ndim - 1) + 1 * (ndim >= 1),
        matrix.shape[1] * (ndim - 2) + 1 * (ndim >= 2),
        matrix.shape[2] * (ndim - 3) + 1 * (ndim >= 3)
    )
