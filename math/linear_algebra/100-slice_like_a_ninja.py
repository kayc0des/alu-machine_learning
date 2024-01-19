#!/usr/bin/env python3
"""
Slice like a ninja
"""


def np_slice(matrix, axes={}):
    """
    Slice like a ninja
    """
    # Create a deep copy of the original matrix to avoid modifying it in place
    # Create a deep copy of the original matrix to avoid modifying it in place
    result = [row.copy() for row in matrix]

    # Apply slices along specified axes
    for axis, slice_tuple in axes.items():
        axis_size = len(result)
        indices = slice(*slice_tuple)

        if indices.start is None:
            indices = slice(0, axis_size, indices.step)
        elif indices.stop is None:
            indices = slice(indices.start, axis_size, indices.step)

        result = [row[indices] for row in result]

    return result

