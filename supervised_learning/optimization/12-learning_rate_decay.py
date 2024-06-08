#!/usr/bin/env python3
''' Creates Training Ops with Adam '''


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''
    Creates a learning rate decay op in tf

    Args:
    alpha: original learning rate
    decay_rate: weight used to the determine the rate
        at which alpha will decay
    global_step: the number of passes of elapsed gradient descent
    decay_step: the number of passes before alpha decays further
    Fashion: stepwise

    Returns:
    The updated value of alpha
    '''

    return tf.train.inverse_time_decay(
        learning_rate=alpha, global_step=global_step, decay_steps=decay_step,
        decay_rate=decay_rate, staircase=True)
