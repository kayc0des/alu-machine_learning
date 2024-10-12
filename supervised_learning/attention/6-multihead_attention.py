#!/usr/bin/env python3
''' Multihead Attention '''


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    Multihead attention
    '''
    def __init__(self, dm, h):
        '''
        Constructor method
        
        Args:
            d_model: int, dimension of the model
            num_heads: int, number of heads
        '''
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)
        
    def split_heads(self, x, batch):
        """
        Splits the last dimension of tensor into (h, dm) and
            transposes the result so the shape is (batch, h, seq_len, dm)
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x
    
    def call(self, Q, K, V, mask):
        '''
        Call method

        Args:
            Q: query tensor of shape (batch, seq_len_q, d_model)
            K: key tensor of shape (batch, seq_len_k, d_model)
            V: value tensor of shape (batch, seq_len_v, d_model)
            mask: mask tensor

        Returns:
            output: output tensor of shape (..., seq_len_q, dm)
            weights: attention weights of shape (..., h, seq_len_q, seq_len_v)
        '''
        # query, key, value
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # split into multiple heads
        Q = self.split_heads(Q, Q.shape[1])
        K = self.split_heads(K, K.shape[1])
        V = self.split_heads(V, V.shape[1])

        # scaled dot product attention
        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)

        # concat attention
        concat_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # output layer
        output = self.linear(concat_attention)

        return output, attention_weights
