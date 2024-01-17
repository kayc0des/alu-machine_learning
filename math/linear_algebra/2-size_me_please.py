#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Determine's the shape of a matrix
    Parameters: matrix: The input matrix
    """
    shape = []
    shape.append(len(matrix))

    def nested_shape(matrix):
        """
        Recursively calculates the shape of nested lists
        Parameters: matrix(list)
        """
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
