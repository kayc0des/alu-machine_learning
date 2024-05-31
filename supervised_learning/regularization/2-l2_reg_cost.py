#!/usr/bin/env python3
''' Calculates the cost with l2 reg '''


import numpy as np
import tensorflow as tf


def l2_reg_cost(cost):
    '''
    Calculates the cost of a neural network

    Param:
    cost -> tensor containing the cost without l2 reg

    Returns:
    Tensor containing l2 reg
    '''

    return cost + tf.losses.get_regularization_losses()
