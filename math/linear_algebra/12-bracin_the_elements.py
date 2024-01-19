#!/usr/bin/env python3
"""
Script defines a fx to calculate element-wise (+, -, *, /)
"""


def np_elementwise(mat1, mat2):
    """
    Function returns element-wise +, -, *, / of two matrices
    """
    # Element-wise addition
    add = mat1 + mat2
    # Element-wise subtraction
    sub = mat1 - mat2
    # Element-wise multiplication
    mul = mat1 * mat2
    # Element-wise division
    div = mat1 / mat2

    return add, sub, mul, div
