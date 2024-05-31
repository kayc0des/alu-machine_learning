#!/usr/bin/env python3
'''
Creates a layer with dropout
'''


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    '''
    create a layer with dropout

    Param:
    prev -> tensor containing output of prev layer
    n -> number of neurons in the layer
    activation -> activation fxn
    keep_prob -> probability that a node will be kept

    Returns:
    output layer
    '''
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    dropout = tf.layers.Dropout(rate=(1 - keep_prob))
    prev_drop = dropout(prev)

    layer = tf.layer.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)
    output = layer(prev_drop)

    return output
