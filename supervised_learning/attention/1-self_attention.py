#!/usr/bin/env python3
""" class SelfAttention that inherits from tensorflow.keras.layers.
    Layer to calculate the attention for machine translation"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ class SelfAttention that inherits from tensorflow.keras.layers.
    Layer to calculate the attention for machine translation"""

    def __init__(self, units):
        """initialize"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculates the attention context vector and
        attention weights"""
        s_prev_expanded = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
