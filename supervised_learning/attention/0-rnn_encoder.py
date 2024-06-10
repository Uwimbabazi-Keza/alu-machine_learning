#!/usr/bin/env python3
"""class RNNEncoder that inherits from tensorflow.keras.layers
Layer to encode for machine translation"""

import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU


class RNNEncoder(tf.keras.layers.Layer):
    """Class constructor def __init__(self, vocab, embedding, units, batch)"""
    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = GRU(units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
