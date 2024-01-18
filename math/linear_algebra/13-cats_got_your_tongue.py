#!/usr/bin/env python3
"""
This script provides a function to concatenate two matrices along a specified axis
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    This function concatenates matrix along a specified axis

    Paramters:
    - mat1 and mat2

    Returns:
    - array: returns a new ndarray
    """
    return np.concatenate((mat1, mat2), axis=axis)
