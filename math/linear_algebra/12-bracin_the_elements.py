#!/usr/bin/env python3
"""
This script provides a function to concatenate two matrices along a specified axis
"""


def np_elementwise(mat1, mat2):
    # Element-wise addition
    add = [[elem1 + elem2 for elem1, elem2 in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
    
    # Element-wise subtraction
    sub = [[elem1 - elem2 for elem1, elem2 in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
    
    # Element-wise multiplication
    mul = [[elem1 * elem2 for elem1, elem2 in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
    
    # Element-wise division
    div = [[elem1 / elem2 for elem1, elem2 in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]

    return add, sub, mul, div
