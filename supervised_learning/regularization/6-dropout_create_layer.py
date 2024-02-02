#!/usr/bin/env python3
"""creates a layer of a neural network using dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a layer of a neural network using dropout
    """
    weight_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    dropout_layer = tf.layers.Dropout(keep_prob)
    layer_model = tf.layers.Dense(units=n, activation=activation,
                                  name="layer", 
                                  kernel_initializer=weight_initializer,
                                  kernel_regularizer=dropout_layer)
    return layer_model(prev)
