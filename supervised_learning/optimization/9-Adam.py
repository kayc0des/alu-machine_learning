#!/usr/bin/env python3
''' Updates variable using RMSProp optimization '''


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''
    Updates variables usinf Adam

    Args:
    alpha: learning rate
    beta1: weight used for the first moment
    beta2: weight used for the second moment
    epsilon: small number to avoid division by zero
    var: numpy.ndarray containing the variable to be updated
    grad: numpy.ndarray containing the gradient of var
    v: previous first moment of var
    s: previous second moment of var
    t: time step used for bias correction

    Returns:
    Updates variable, first and second new moment
    '''

    v = (beta1 * v) + ((1 - beta1) * grad)
    s = (beta2 * s) + ((1 - beta2) * np.square(grad))

    v_corrected = v / (1 - np.power(beta1, t))
    s_corrected = s / (1 - np.power(beta2, t))

    var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, v_corrected, s_corrected
