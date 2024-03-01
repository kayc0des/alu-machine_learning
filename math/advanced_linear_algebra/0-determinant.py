#!/usr/bin/env python3

''' This function returns the determinant of a matrix '''

def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The determinant of the matrix.

    Raises:
        TypeError: If the input is not a list of lists.
        ValueError: If the matrix is not square.
    """

    # Check for valid input type
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check for square matrix
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases:
    if n == 0:  # 0x0 matrix
        return 1
    elif n == 1:  # 1x1 matrix
        return matrix[0][0]

    # Recursive calculation using cofactor expansion
    det = 0
    for i in range(n):
        det += (-1) ** i * matrix[0][i] * determinant(minor(matrix, 0, i))
    return det

def minor(matrix, i, j):
    """
    Calculates the minor matrix by removing row i and column j.
    """
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
