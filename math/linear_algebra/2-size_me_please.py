#!/usr/bin/env python3
def matrix_shape(matrix):
    shape = []
    shape.append(len(matrix))

    def nested_shape(matrix):
        get_nested_shape = set()
        for sub_list in matrix:
            if isinstance(sub_list, list):
                num_elements = len(sub_list)
                get_nested_shape.add(num_elements)
                if isinstance(sub_list[0], list):
                    get_nested_shape |= nested_shape(sub_list)
            else:
                # Handle the case where sub_list is not a list (e.g., an integer)
                num_elements = 1
                get_nested_shape.add(num_elements)
                
        return get_nested_shape
        
    sub_shape = list(nested_shape(matrix))
    shape.extend(sub_shape)
    return shape
