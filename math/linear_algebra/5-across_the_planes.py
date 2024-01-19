#!/usr/bin/env python3
"""
This script defines a function that adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Function adds two matrices element-wise
    """
    mat3 = []
    if len(mat1[0]) == len(mat2[0]):
        for i in range(len(mat1)):
            temp_mat1 = mat1[i]
            temp_mat2 = mat2[i]
            inner_element = []
            for j in range(len(temp_mat1)):
                inner_element.append(temp_mat1[j] + temp_mat2[j])
            mat3.append(inner_element)
    else:
        return None
    return mat3
