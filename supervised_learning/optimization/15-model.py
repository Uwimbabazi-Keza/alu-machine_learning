#!/usr/bin/env python3

"""
builds, trains, and saves a neural network model in tensorflow using Adam optimization,
mini-batch gradient descent, learning rate decay, and batch normalization
"""

import tensorflow as tf
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    builds, trains, and saves a neural network model in tensorflow using Adam optimization,
    mini-batch gradient descent, learning rate decay, and batch normalization
    """

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='x')
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name='y')

    A = create_layer(x, layers[0], activations[0])
    for i in range(1, len(layers)):
        A = create_batch_norm_layer(A, layers[i], activations[i])
    y_pred = A

    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    global_step = tf.Variable(0, trainable=False)
    alpha = tf.train.inverse_time_decay(
        alpha, global_step, decay_rate, 1, staircase=True)

    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs + 1):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            session.run(train_op, feed_dict={x: X_train, y: Y_train})
            if epoch % 10 == 0:
                train_cost = session.run(
                    loss, feed_dict={x: X_train, y: Y_train})
                valid_cost = session.run(
                    loss, feed_dict={x: X_valid, y: Y_valid})
                train_accuracy = session.run(
                    accuracy, feed_dict={x: X_train, y: Y_train})
                valid_accuracy = session.run(
                    accuracy, feed_dict={x: X_valid, y: Y_valid})
                print("After {} epochs:".format(epoch))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))
        saves = saver.save(session, save_path)
    
    return saves
