#!/usr/bin/env python3
''' SDP Attention '''


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    '''
    SDP Attention

    Args:
        Q: query matrix
        K: key matrix
        V: value matrix
    '''
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(K)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to the scaled tensor
    if mask is not None:
        logits += mask * -1e9

    # apply softmax to the scaled logits
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, V)

    return output, attention_weights
