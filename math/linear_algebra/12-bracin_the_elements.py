#!/usr/bin/env python3
"""
This script provides a function to concatenate two matrices along a specified axis
"""


def np_elementwise(mat1, mat2):
    def elementwise_op(func, matrix1, matrix2):
        return [
            [func(e1, e2) for e1, e2 in zip(row1, row2)]
            for row1, row2 in zip(matrix1, matrix2)
        ]

    return (
        elementwise_op(lambda x, y: x + y, mat1, mat2),
        elementwise_op(lambda x, y: x - y, mat1, mat2),
        elementwise_op(lambda x, y: x * y, mat1, mat2),
        elementwise_op(lambda x, y: x / y, mat1, mat2),
    )
