#!/usr/bin/env python3
"""
Slice like a ninja
"""


import numpy as np


def np_slice(matrix, axes={}):
    """
    Slice like a ninja
    """
    # Create a deep copy of the original matrix to avoid modifying it in place
    result = matrix.copy()

    # Apply slices along specified axes
    for axis, slice_tuple in axes.items():
        result = np.take(result, slice(*slice_tuple), axis=axis)

    return result

