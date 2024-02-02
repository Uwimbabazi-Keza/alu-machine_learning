#!/usr/bin/env python3
"""function  that creates a tensorflow layer
that includes L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ that creates a tensorflow layer
    that includes L2 regularization
    """

    regularizer = tf.keras.regularizers.l2(lambtha)
    weight = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        name="layer",
        kernel_initializer=weight,
        kernel_regularizer=regularizer
    )
    
    return layer(prev)