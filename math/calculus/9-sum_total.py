#!/usr/bin/env python3
'''
    A function that returns the summation of i**2
'''


def summation_i_squared(n):
    """
    Returns the summation of i**2
    for values of i from 0 to n
    """

    if type(n) == int:
        return n * (n + 1) * (2 * n + 1) // 6
    elif type(n) != 'int':
        return None
