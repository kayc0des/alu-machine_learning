#!/usr/bin/env python3
'''
Creates a layer with dropout
'''


import tensorflow as tf
import numpy as np

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

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)
    output = layer(prev)
    layer = tf.layers.dropout(output, rate=1-keep_prob)
    return layer

if __name__ == '__main__':
    tf.set_random_seed(0)
    np.random.seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    X = np.random.randint(0, 256, size=(10, 784))
    a = dropout_create_layer(x, 256, tf.nn.tanh, 0.8)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a, feed_dict={x: X}))
