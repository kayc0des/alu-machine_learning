#!/usr/bin/env python3
''' Creates layer with batch nor, '''


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''
    creates a batch normalization layer for
    a neural network in tensorflow:

    Args:
    prev -> activated output of the previous layer
    n -> number of nodes in the layer to be created
    activation -> the activation function that should be used
    gamma and beta, initialized as vectors of 1 and 0 respectively

    Returns:
    Activated output of the layer
    '''

    # weight initializer
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # layer
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = layer(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-8)

    return activation(Z_norm)
