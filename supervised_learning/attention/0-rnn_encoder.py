#!/usr/bin/env python3
"""class RNNEncoder that inherits from tensorflow.keras.layers
Layer to encode for machine translation"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """class RNNEncoder that inherits from tensorflow.keras.layers
    Layer to encode for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """initialize"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """initialize hidden state"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Calculates the attention context vector and
        attention weights"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
