#!/usr/bin/env python3
"""
This script calculates a correlation matrix
"""


# import necessary modules
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix

    Args:
        C is a numpy.ndarray containing a correlation matrix

    Returns:
        A numpy.ndarray containing the correlation matrix

    Raise:
        TypeError: If C is not a numpy.ndarray
        ValueError: If C does not have shape(d, d)
    """
    # run checks
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')

    d = C.ndim

    if C.shape[0] != d or C.shape[1] != d:
        raise ValueError('C must be a 2D square matrix')

    correlation_matrix = np.corrcoef(C, rowvar=False)
    return correlation_matrix

