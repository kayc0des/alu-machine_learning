#!/usr/bin/env python3
"""
This script defines a fn that recursively concatenates two matrices
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Function recursively concatenates two matrices
    """
    def recursive_concat(m1, m2, current_axis):
        if isinstance(m1, list) and isinstance(m2, list):
            if len(m1) != len(m2):
                return None
            if current_axis == axis:
                return m1 + m2
            result = []
            for i in range(len(m1)):
                sub_result = recursive_concat(m1[i], m2[i], current_axis + 1)
                if sub_result is None:
                    return None
                result.append(sub_result)
            return result
        elif current_axis == axis:
            return None
        else:
            return m1 + m2

    result = recursive_concat(mat1, mat2, 0)
    return result if result is not None else None
