#!/usr/bin/env python3
''' Creates Training Ops with Adam '''


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''
    Creates Training op using GD with momentum

    Args:
    loss -> loss of the network
    alpha -> learning rate
    beta1 -> weight for first moment
    beta2 -> weight for second moment
    epsilon -> small number to avoid zero division

    Returns:
    Adam optimization operation
    '''

    return tf.train.MomentumOptimizer(
        alpha, beta1, beta2, epsilon).minimize(loss)
