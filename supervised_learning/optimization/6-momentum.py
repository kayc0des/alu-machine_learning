#!/usr/bin/env python3
''' Creates Training Ops with momentum '''


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    '''
    Creates Training op using GD with momentum

    Args:
    loss -> loss of the network
    alpha -> learning rate
    beta1 -> momentum weight

    Returns:
    Momentum optimization operation
    '''
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
