#!/usr/bin/env python3

''' This function returns the definiteness of a matrix '''

import numpy as np


def definiteness(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.shape[0] != matrix.shape[1]:
        return None

    eigvals = np.linalg.eigvals(matrix)
    positive_count = np.sum(eigvals > 0)
    negative_count = np.sum(eigvals < 0)
    zero_count = np.sum(eigvals == 0)

    if zero_count > 0:
        return "Indefinite"
    elif negative_count == 0:
        return "Positive definite" if positive_count == matrix.shape[0] else "Positive semi-definite"
    elif positive_count == 0:
        return "Negative definite" if negative_count == matrix.shape[0] else "Negative semi-definite"
    else:
        return None
