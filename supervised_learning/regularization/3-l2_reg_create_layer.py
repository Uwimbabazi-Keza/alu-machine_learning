#!/usr/bin/env python3
"""function  that creates a tensorflow layer
that includes L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ that creates a tensorflow layer
    that includes L2 regularization
    """

    initializer = tf.keras.initializers.glorot_uniform()
    regularizer = tf.keras.regularizers.l2(lambtha)
    weights = tf.Variable(initializer(shape=(prev.shape[1], n)), name="kernel", regularizer=regularizer)

    
    biases = tf.Variable(tf.zeros((1, n)), name="bias")

    z = tf.matmul(prev, weights) + biases

    if activation is not None:
        output = activation(z)
    else:
        output = z

    return output
