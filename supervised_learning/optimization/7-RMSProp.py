#!/usr/bin/env python3
''' Updates variable using RMSProp optimization '''


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''
    Updates a variable using RMSProp algorithm

    Args:
    alpha -> learning rate
    beta2 -> RMSProp weight
    epsilon -> small number to avoid division by zero
    var -> numpy.ndarray containing the variable to be updated
    grad -> numpy.ndarray containing the gradient of var
    s -> previous second moment of var

    Returns:
    The updated variable and the new moment
    '''

    # update gradients
    s = (beta2 * s) + ((1 - beta2) * np.square(grad))

    # update variables
    var = var - alpha * (grad / (np.sqrt(s) + epsilon))

    return var, s
