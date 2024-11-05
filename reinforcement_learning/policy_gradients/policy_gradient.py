#!/usr/bin/env python3
''' This script computes a policy '''


import numpy as np


def policy(matrix, weight):
    """
    Computes a policy by applying a weight to each element in a matrix.

    Parameters:
    matrix (2D array-like): The matrix to which the weight is applied.
    weight (float): The weight factor to apply to the matrix.

    Returns:
    numpy.ndarray: A new matrix where each element is the original matrix element 
                   multiplied by the weight.
    """
    # Convert matrix to a numpy array for element-wise operations
    matrix = np.array(matrix)
    
    # Apply the weight to the matrix
    weighted_matrix = matrix @ weight
    
    return weighted_matrix


''' Debug '''
if __name__ == '__main__':
    weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01], 
    [1.14374817e-04, 3.02332573e-01], 
    [1.46755891e-01, 9.23385948e-02], 
    [1.86260211e-01, 3.45560727e-01]
    ]))
    state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214,  0.01636746,  0.01196594, -0.03095031]
    ]))

    res = policy(state, weight)
    print(res)