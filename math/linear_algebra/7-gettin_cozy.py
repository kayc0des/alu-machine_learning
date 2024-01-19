#!/usr/bin/env python3
"""
Module to concatenate 2D matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Args:
        - mat1: First matrix
        - mat2: Second matrix
        - axis: Axis along which to concatenate (0 for rows, 1 for columns)

    Returns:
        - New matrix if concatenation is possible, None otherwise
    """

    # Check if concatenation is possible
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    elif axis == 1 and len(mat1) != len(mat2):
        return None

    # Perform concatenation
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
