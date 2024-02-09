#!/usr/bin/env python3

"""
builds, trains, and saves a neural network model in tensorflow using Adam optimization,
mini-batch gradient descent, learning rate decay, and batch normalization
"""
import tensorflow as tf
import numpy as np


def create_p(nx, classes):
    return tf.placeholder(float, shape=[None, nx], name='x'), tf.placeholder(
        float, shape=[None, classes], name='y')


def create_l(prev, n, activation):
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name="layer")
    return layer(prev)


def f_p(x, layer_sizes=[], activations=[]):
    prediction = batch_norm_layer(x, layer_sizes[0], activations[0])
    for layer, activation in zip(layer_sizes[1:], activations[1:]):
        prediction = batch_norm_layer(prediction, layer, activation)
    return prediction


def cal_a(y, y_pred):
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    mean = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    return mean


def cal_l(y, y_pred):
    return tf.losses.softmax_cross_entropy(y, y_pred)


def shuffle(X, Y):
    permutation = np.random.permutation(X.shape[0])
    return X[permutation, :], Y[permutation, :]


def adam_op(loss, alpha, beta1, beta2, epsilon):
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def batch_norm_layer(prev, n, activation):
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, kernel_initializer=k_init)
    Z = output(prev)

    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]), name="beta")

    mean, var = tf.nn.moments(Z, axes=0)
    b_norm = tf.nn.batch_normalization(Z, mean, var, offset=beta,
                                       scale=gamma,
                                       variance_epsilon=1e-8)
    if activation is None:
        return b_norm
    else:
        return activation(b_norm)


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):

    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x, y = create_p(nx, classes)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    y_pred = f_p(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)

    accuracy = cal_a(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)

    loss = cal_l(y, y_pred)
    tf.add_to_collection("loss", loss)

    global_step = tf.Variable(0)
    alpha_d = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = adam_op(loss, alpha_d, beta1, beta2, epsilon)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        m = X_train.shape[0]
        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = m // batch_size + 1

        for i in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))

            if i < epochs:
                shuffled_X, shuffled_Y = shuffle(X_train, Y_train)

                for b in range(n_batches):
                    start = b * batch_size
                    end = (b + 1) * batch_size
                    if end > m:
                        end = m
                    X_mini_batch = shuffled_X[start:end]
                    Y_mini_batch = shuffled_Y[start:end]

                    next_train = {x: X_mini_batch, y: Y_mini_batch}
                    sess.run(train_op, feed_dict=next_train)

                    if (b + 1) % 100 == 0 and b != 0:
                        loss_mini_batch = sess.run(loss, feed_dict=next_train)
                        acc_mini_batch = sess.run(accuracy,
                                                  feed_dict=next_train)
                        print("\tStep {}:".format(b + 1))
                        print("\t\tCost: {}".format(loss_mini_batch))
                        print("\t\tAccuracy: {}".format(acc_mini_batch))

            sess.run(tf.assign(global_step, global_step + 1))

        return saver.save(sess, save_path)

