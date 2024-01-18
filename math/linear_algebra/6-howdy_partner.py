#!/usr/bin/env python3
"""
Defines a fxn that concatenates two matrices
"""


def cat_arrays(arr1, arr2):
    """
    Returns the a concatenated array of both matrices
    """
    arr3 = arr1.extend(arr2)
    return arr3
