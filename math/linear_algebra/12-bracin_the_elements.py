#!/usr/bin/env python3
"""
This script provides a function to concatenate two matrices along a specified axis
"""


def np_elementwise(mat1, mat2):
    is_list = lambda x: isinstance(x, list)

    add = lambda x, y: x + y
    sub = lambda x, y: x - y
    mul = lambda x, y: x * y
    div = lambda x, y: x / y

    add_result = list(map(add, np_elementwise(mat1[0], mat2[0]))) if is_list(mat1) and is_list(mat2) else mat1 + mat2
    sub_result = list(map(sub, np_elementwise(mat1[0], mat2[0]))) if is_list(mat1) and is_list(mat2) else mat1 - mat2
    mul_result = list(map(mul, np_elementwise(mat1[0], mat2[0]))) if is_list(mat1) and is_list(mat2) else mat1 * mat2
    div_result = list(map(div, np_elementwise(mat1[0], mat2[0]))) if is_list(mat1) and is_list(mat2) else mat1 / mat2

    return add_result, sub_result, mul_result, div_result
