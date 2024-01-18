#!/usr/bin/env python3
"""
This script provides a function to evaluate the transpose of a matrix
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    This function returns the transpose of a matrix

    Paramters:
    - matrix: an ndarray

    Returns:
    - array: returns a new ndarray
    """
    return np.concatenate((mat1, mat2), axis=0)
