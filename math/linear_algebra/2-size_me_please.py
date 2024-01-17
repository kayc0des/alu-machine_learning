#!/usr/bin/env python3
def matrix_shape(matrix):
    """ Define's the shape of a matrix"""
    shape = []
    shape.append(len(matrix))

    def nested_shape(matrix):
        """ Get shape of nested List"""
        get_nested_shape = set()
        for sub_list in matrix:
            if isinstance(sub_list, list):
                num_elements = len(sub_list)
                get_nested_shape.add(num_elements)
                if isinstance(sub_list[0], list):
                    get_nested_shape |= nested_shape(sub_list)         
        return get_nested_shape
    sub_shape = list(nested_shape(matrix))
    shape.extend(sub_shape)
    return shape
