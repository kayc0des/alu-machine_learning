#!/usr/bin/env python3
''' Updates variable using gd momentum '''


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''
    Updates a variable using the gradient
    descent with momentum optimization algorithm

    Args:
    alpha -> the learning rate
    beta1 -> the momentum weight
    var -> numpy.ndarray containing the variable to be updated
    grad -> numpy.ndarray containing the gradient of var
    v -> previous first moment of var

    Returns:
    The updated variable and the new moment
    '''
    m = var.shape[0]
    v_prev = v
    updated_var = np.zeros((m, var.shape[1]))

    for i in range(m):
        v_prev = (beta1 * v_prev) + ((1 - beta1) * grad[i])
        val = var[i] - (alpha * v_prev)
        updated_var[i] = val
        
    return updated_var, v_prev
