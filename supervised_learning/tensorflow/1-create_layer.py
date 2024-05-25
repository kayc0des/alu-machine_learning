#!/usr/bin/env python3
''' Creates layer '''


import tensorflow as tf


def create_layer(prev, n, activation):
    '''
    Creates layers
    Returns: the tensor output of the layer
    '''
    # Define the variance scaling initializer with mode "FAN_AVG"
    initializer = tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG")

    # Create the layer with the specified initializer and activation function
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer, name='layer')

    # Compute the output of the layer
    output = layer(prev)

    return output
