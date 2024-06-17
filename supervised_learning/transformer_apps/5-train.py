#!/usr/bin/env python3
"""def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs): that creates and trains a transformer model for machine translation of Portuguese to
English using our previously created dataset"""

import tensorflow as tf
import numpy as np
from 3-dataset import Dataset
from 4-create_masks import create_masks
from 5-transformer import Transformer

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, tf.shape(y_true)[2]))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=loss.dtype)
    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return loss

def accuracy_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, tf.shape(y_true)[2]))
    accuracies = tf.equal(y_true, tf.argmax(y_pred, axis=2))
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    accuracies = tf.reduce_sum(accuracies * mask) / tf.reduce_sum(mask)
    return accuracies

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dm, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.dm = dm
        self.dm = tf.cast(self.dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)

def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    dataset = Dataset(batch_size, max_len)
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    transformer = Transformer(N, dm, h, hidden, dataset.tokenizer_pt.vocab_size + 2,
                              dataset.tokenizer_en.vocab_size + 2, max_len, max_len)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function
    def train_step(inputs, targets):
        enc_mask, combined_mask, dec_mask = create_masks(inputs, targets)

        with tf.GradientTape() as tape:
            predictions = transformer(inputs, targets, True, enc_mask, combined_mask, dec_mask)
            loss = loss_function(targets[:, 1:], predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(targets[:, 1:], predictions))

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inputs, targets)) in enumerate(dataset.data_train):
            train_step(inputs, targets)

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1}, batch {batch}: loss {train_loss.result():.4f}, accuracy {train_accuracy.result():.4f}')

        print(f'Epoch {epoch + 1}: loss {train_loss.result():.4f}, accuracy {train_accuracy.result():.4f}')

    return transformer
