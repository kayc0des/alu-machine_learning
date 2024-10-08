#!/usr/bin/env python3
''' Calculates the attention of a machine translation '''


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    '''
    This class calculates the attention of
    a machine translation
    '''

    def __init__(self, units):
        '''
        Constuctor method

        Args:
            units -

        Returns:
            None
        '''
        super(SelfAttention, self).__init__()

        # public instance attributes
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        '''
        Call method

        Args:
            s_prev - a tensor of shape (batch, units)
            hidden_states - tensor of shape (batch, input_sq_len,
            units) conatining the outputs of the encoder

        Returns:
            Context - a tensor of shape (batch, units)
                containing the context vector of the decoder
            weights - a tensor of shape (batch, input_seq_len, 1)
                containing the attention weights
        '''

        s_prev = tf.expand_dims(s_prev, 1)

        # Apply dense layers to prev decoder state & encoder hidden states
        W_s = self.W(s_prev)
        U_h = self.U(hidden_states)

        # Calculate score by applying tanh to the sum of W_s and U_h
        score = tf.nn.tanh(W_s + U_h)

        # Apply V to the tanh result to get attention scores
        attention_weights = self.V(score)

        # Remove the last dim & apply softmax to get attention distribution
        attention_weights = tf.nn.softmax(attention_weights, axis=1)

        # Compute the context vec as the 
        # weighted sum of the encoder hidden states
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
