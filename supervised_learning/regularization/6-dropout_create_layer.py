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
    layer = tf.layers.dense(prev, n)
    layer = activation(layer)
    layer = tf.nn.dropout(layer, keep_prob)

    return layer
