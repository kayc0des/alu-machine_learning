#!/usr/bin/env python3
''' Putting it all together '''

import tensorflow as tf
import numpy as np


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    '''
    Builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization

    Args:
    Data_train: tuple containing the training inputs and training labels, respectively
    Data_valid: tuple containing the validation inputs and validation labels, respectively
    layers: list containing the number of nodes in each layer of the network
    activations: list containing the activation functions used for each layer of the network
    alpha: learning rate
    beta1: weight for the first moment of Adam Optimization
    beta2: weight for the second moment of Adam Optimization
    epsilon: small number used to avoid division by zero
    decay_rate: decay rate for inverse time decay of the learning rate (decay step should be 1)
    batch_size: number of data points that should be in a mini-batch
    epochs: number of times the training should pass through the whole dataset
    save_path: path where the model should be saved to

    Returns:
    The path where the model was saved
    '''

    def create_placeholders(nx, classes):
        '''Create the placeholders for the tensorflow session'''
        x = tf.placeholder(tf.float32, [None, nx], name='x')
        y = tf.placeholder(tf.float32, [None, classes], name='y')
        return x, y

    def forward_propagation(x, layer_sizes, activations):
        '''Creates the forward propagation graph for the neural network'''
        for i in range(len(layer_sizes)):
            if i == 0:
                layer = tf.layers.dense(x, units=layer_sizes[i], activation=None)
            else:
                layer = tf.layers.dense(prev_layer, units=layer_sizes[i], activation=None)

            layer = tf.layers.batch_normalization(layer)
            layer = activations[i](layer)
            prev_layer = layer
        return prev_layer

    def shuffle_data(X, Y):
        '''Shuffles the data points in two matrices the same way'''
        permutation = np.random.permutation(X.shape[0])
        return X[permutation], Y[permutation]

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_propagation(x, layers, activations)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), tf.float32))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(alpha, global_step, 1, decay_rate, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        num_batches = X_train.shape[0] // batch_size + (X_train.shape[0] % batch_size != 0)

        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)

            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size

                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                _, step_cost, step_accuracy = sess.run([train_op, loss, accuracy], feed_dict={x: X_batch, y: Y_batch})

                if (batch + 1) % 100 == 0:
                    print(f"\tStep {batch + 1}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print(f"After {epoch + 1} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

        save_path = saver.save(sess, save_path)
    
    return save_path
