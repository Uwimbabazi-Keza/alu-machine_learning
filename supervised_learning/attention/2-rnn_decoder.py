#!/usr/bin/env python3
"""a class RNNDecoder that inherits from tensorflow.keras.layers.
Layer to decode for machine translation"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """a class RNNDecoder that inherits from tensorflow.keras.layers.
    Layer to decode for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """initialize"""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        
    def call(self, x, s_prev, hidden_states):
        """Decode for machine translation"""
        x = self.embedding(x)
        
        s_prev_expanded = tf.expand_dims(s_prev, 1)
        score = tf.keras.layers.Dense(1)(tf.nn.tanh(s_prev_expanded + hidden_states))
        weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(weights * hidden_states, axis=1)
        
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        y, s = self.gru(x, initial_state=s_prev)
        
        y = self.F(y)
        
        return y, s
