#!/usr/bin/env python3

"""
creates a batch normalization layer for a neural
network in tensorflow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural
    network in tensorflow
    """
    k = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, kernel_initializer=k)
    z = model(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    mean, variance = tf.nn.moments(z, axes=[0])
    norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma, 1e-8)
    return activation(norm)
