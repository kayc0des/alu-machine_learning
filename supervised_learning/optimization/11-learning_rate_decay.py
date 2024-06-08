#!/usr/bin/env python3
''' Updates variable using RMSProp optimization '''


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''
    Updates learning rate using inverse time decay

    Args:
    alpha: original learning rate
    decay_rate: weight used to the determine the rate
        at which alpha will decay
    global_step: the number of passes of elapsed gradient descent
    decay_step: the number of passes before alpha decays further
    Fashion: stepwise

    Returns:
    The updated value of alpha'''
    
    alpha = alpha / (1 + decay_rate * (global_step / decay_step))

    return alpha
