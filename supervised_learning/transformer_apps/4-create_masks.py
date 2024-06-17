#!/usr/bin/env python3
"""Create the function def create_masks(inputs, target):
that creates all masks for training/validation"""

import tensorflow as tf

def create_padding_mask(seq):
    """Creates a padding mask for the given sequence"""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    """Creates a look ahead mask to mask future tokens in a sequence"""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inputs, target):
    """Creates all masks for training/validation"""
    encoder_mask = create_padding_mask(inputs)
    decoder_padding_mask = create_padding_mask(inputs)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    decoder_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)
    return encoder_mask, combined_mask, decoder_padding_mask
