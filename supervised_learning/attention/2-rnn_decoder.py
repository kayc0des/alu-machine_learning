#!/usr/bin/env python3
''' RNN Decoder '''


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    ''' RNN Decoder '''

    def __init__(self, vocab, embedding, units, batch):
        '''
        Class constructor

        Args:
            vocab - int representing the size of the output vocabulary
            embedding - Embedding layer used to embed the inputs
            units - int representing the number of hidden units in the RNN
            batch - int representing the batch size
        '''
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.units = units
        self.batch = batch
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
                                  
    def call(self, x, s_prev, hidden_states):
        '''
        Method to call the layer

        Args:
            x - tensor of shape (batch, 1) containing the previous
                decoder hidden state
            s_prev - tensor of shape (batch, units) containing the previous
                decoder hidden state
            hidden_states - tensor of shape (batch, input_seq_len, 2 * units)
                containing the outputs of the encoder

        Returns:
            outputs - tensor of shape (batch, vocab) containing the outputs
                of the decoder
            s - tensor of shape (batch, units) 
                containing the new decoder hidden state
        '''
        # Embed the input
        x = self.embedding(x)

        # Remove the extra dimension from x
        x = tf.squeeze(x, axis=1)

        # Concatenate the input and the
        # previous decoder hidden state
        attention = SelfAttention(self.units)
        context, attention_weights = attention.call(s_prev, hidden_states)
        x = tf.concat([x, context], axis=-1)

        # Pass the concatenated input through the GRU
        output, s = self.gru(tf.expand_dims(x, 1), initial_state=s_prev)

        # Remove the extra dimension from output
        output = tf.squeeze(output, axis=1)

        # Pass the output through the dense layer
        output = self.F(output)

        return output, s
