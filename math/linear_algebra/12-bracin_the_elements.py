#!/usr/bin/env python3
"""
This script provides a function to concatenate two matrices along a specified axis
"""


def np_elementwise(mat1, mat2):
    # Element-wise addition
    add = mat1 + mat2
    
    # Element-wise subtraction
    sub = mat1 - mat2
    
    # Element-wise multiplication
    mul = mat1 * mat2
    
    # Element-wise division
    div = mat1 / mat2

    return add, sub, mul, div
