#!/usr/bin/env python3
''' RNNEncoder Class '''


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    '''
    Inherits from tf.keras.Layers to encode for machine learning
    '''

    def __init__(self, vocab, embedding, units, batch):
        '''
        Constructor method

        Args:
            vocab - int representing size of input vocabulary
            embedding - int representing dim. of embedding vector
            units - int representing the number of hidden units in RNN Cell
            batch - int representing batch size

        Returns:
            None
        '''
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab,
                                                   output_dim = embedding)
        self.gru = tf.keras.layers.GRU(units = units,
                       recurrent_initializer = 'glorot_uniform',
                       return_sequences = True,
                       return_state = True)
        
    def initialize_hidden_state(self):
        '''
        Initializes the hidded states of the
        RNN Cell to tensor of zeros

        Args:
            None
   
        Returns:
            A tensor of shape (batch, units)
        '''
        return tf.zeros(shape=(self.batch, self.units))
    
    def call(self, x, initial):
        '''
        Call function
    
        Args:
            x - a tensor of shape (batch, input_seq_len)
            initial - tensor of shape (batch, units)
    
        Returns:
            outputs - a tensor of shape (batch,
            input_seq_len, units) containing encoder outputs
            hiddden - a tensor of shape (batch,
            units) comtaining the last hidden state of the encoder
        '''

        # pass input through the embedding layer
        x = self.embedding(x)

        output, state = self.gru(x, initial_state=initial)

        return output, state
