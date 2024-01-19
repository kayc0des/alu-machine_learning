#!/usr/bin/env python3
"""
This script defines a function that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Function adds two arrays element-wise
    """
    arr3 = []

    for i in range(len(arr1)):
        arr3.append(arr1[i] + arr2[i])

    return arr3
