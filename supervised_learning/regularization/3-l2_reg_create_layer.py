#!/usr/bin/env python3
''' Create a layer with l2 reg '''


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''
    Creates a layer with l2 reg

    Param:
    prev -> tensor containing output of prev layer
    n -> number nodes in the new layer
    activation -> activation function
    lambtha -> reg parameter

    Returns:
    the output of the new layer
    '''
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    layer = tf.layers.Dense(n, activation = activation, kernel_regularizer = regularizer) 
    output = layer(prev)

    return output
