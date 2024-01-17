#!/usr/bin/env python3
"""
This script provides a function to evaluate matrix multiplication
"""


import numpy as np

def np_matmul(mat1, mat2):
    """
    Function to evaluate matrix multiplication
    Different from element wise multiplication

    Parameters:
    - mat1, mat2: nnumpy n-dimensional arrays

    Returns:
    - matrix: A new matrix equal to mat1 @ mat 2
    """
    return mat1 @ mat2
