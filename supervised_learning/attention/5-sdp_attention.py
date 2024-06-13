#!/usr/bin/env python3
"""calculates the scaled
dot product attention"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """calculates the scaled
    dot product attention"""
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights
