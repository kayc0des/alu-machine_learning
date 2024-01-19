#!/usr/bin/env python3
"""
This script provides a function to evaluate the transpose of a matrix
"""


def np_transpose(matrix):
    """
    This function returns the transpose of a matrix

    Paramters:
    - matrix: an ndarray

    Returns:
    - array: returns a new ndarray
    """
    def helper(matrix, i, j):
        return [] if i >= len(matrix[0]) else [matrix[j][i]] + helper(matrix, i, j + 1)

    return [] if not matrix else [helper(matrix, i, 0) for i in range(len(matrix[0]))]
