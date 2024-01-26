#!/usr/bin/env python3
"""Functiton def create_layer(prev, n, activation)
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """def create_layer(prev, n, activation)"""
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name='layer')
    output = layer(prev)

    return output
