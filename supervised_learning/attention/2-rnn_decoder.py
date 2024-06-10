#!/usr/bin/env python3
"""a class RNNDecoder that inherits from tensorflow.keras.layers.
Layer to decode for machine translation"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, GRU, Dense
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(Layer):
    """a class RNNDecoder that inherits from tensorflow.keras.layers.
    Layer to decode for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        """initialize"""
        super(RNNDecoder, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding
        self.units = units
        self.batch = batch
        self.embedding = Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = GRU(units, return_sequences=True, return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.F = Dense(vocab)
        self.attention = SelfAttention(units)
        
    def call(self, x, s_prev, hidden_states):
        """Decode for machine translation"""
        x = self.embedding(x)
        context_vector, attention_weights = self.attention(s_prev, hidden_states)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x, initial_state=s_prev)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        
        return y, state
