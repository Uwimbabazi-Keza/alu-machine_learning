#!/usr/bin/env python3
"""creates a layer of a neural network using dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a layer of a neural network using dropout
    """
    initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    
    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)
    
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        name="layer",
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)  # You can adjust the regularization strength
    )
    
    output = dropout_layer(layer(prev))

    return output
