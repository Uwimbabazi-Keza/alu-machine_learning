#!/usr/bin/env python3
"""Function def calculate_accuracy(y, y_pred)
that calculates the accuracy of a prediction
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction
    """

    correct_predictions = tf.equal(tf.argmax(y, axis=1),
                                   tf.argmax(y_pred, axis=1))

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
