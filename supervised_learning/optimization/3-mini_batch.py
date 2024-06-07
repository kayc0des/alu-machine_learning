#!/usr/bin/env python3
''' Trains a loaded neural network '''


import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
    batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    '''
    Trains a loaded neaural network model using mini batch

    Args:
    X_train -> (m, 784) training data
    Y_train -> (m, 10) training labels
    X_valid -> (m, 784) validation data
    Y_valid -> (m, 10) validation labels
    batch_size is the number of 'm' in a batch
    epochs -> # of iterations
    load_path -> path from which to load a model
    save_path -> path to save model after training

    Returns:
    Path to where the model was saved
    '''
    saver = tf.train.import_meta_graph(save_path)
