#!/usr/bin/env python3
"""
This script provides a function to concatenate two matrices along a specified axis
"""


def np_elementwise(mat1, mat2):
    if isinstance(mat1, list) and isinstance(mat2, list):
        # Element-wise addition
        add = np_elementwise(mat1[0], mat2[0])
        
        # Element-wise subtraction
        sub = np_elementwise(mat1[0], mat2[0])
        
        # Element-wise multiplication
        mul = np_elementwise(mat1[0], mat2[0])
        
        # Element-wise division
        div = np_elementwise(mat1[0], mat2[0])

        return add, sub, mul, div

    else:
        # Base case: elements are not lists, perform the operation
        add = mat1 + mat2
        sub = mat1 - mat2
        mul = mat1 * mat2
        div = mat1 / mat2

        return add, sub, mul, div
