#!/usr/bin/env python3
''' Creates a placeholder '''


import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer in a neural network

    Parameters:
    prev -- the tensor output of the previous layer
    n -- the number of nodes in the layer to create
    activation -- the activation function that the layer should use

    Returns:
    the tensor output of the layer
    """
    # Define the initializer using He et. al initialization
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Create the layer with the specified number of nodes and activation function
    layer = tf.layers.dense(inputs=prev, units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')

    return layer

