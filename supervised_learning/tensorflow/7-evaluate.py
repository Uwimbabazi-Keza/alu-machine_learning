#!/usr/bin/env python3
"""def evaluate(X, Y, save_path):
that evaluates the output of a neural network
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    evaluates the output of a neural network.
    """

    tf.reset_default_graph()

    saver = tf.train.import_meta_graph(save_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        eval_loss, eval_accuracy, y_pred_eval = sess.run(
            [loss, accuracy, y_pred], feed_dict={x: X, y: Y}
        )

    return y_pred_eval, eval_accuracy, eval_loss
