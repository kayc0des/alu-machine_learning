#!/usr/bin/env python3
"""
Slice like a ninja
"""


def add_matrices(mat1, mat2):
    def recursive_add(m1, m2):
        if isinstance(m1, list) and isinstance(m2, list):
            if len(m1) != len(m2):
                return None
            result = []
            for i in range(len(m1)):
                sub_result = recursive_add(m1[i], m2[i])
                if sub_result is None:
                    return None
                result.append(sub_result)
            return result
        elif isinstance(m1, (int, float)) and isinstance(m2, (int, float)):
            return m1 + m2
        else:
            return None

    result = recursive_add(mat1, mat2)
    return result