#!/usr/bin/env python3
"""
Defines a fxn that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Function returns the transpose of a matrix
    """
    transpose = []
    for col in range(len(matrix[0])):
        temp_matrix = []
        for row in range(len(matrix)):
            temp_matrix.append(matrix[row][col])
        transpose.append(temp_matrix)
    return transpose
