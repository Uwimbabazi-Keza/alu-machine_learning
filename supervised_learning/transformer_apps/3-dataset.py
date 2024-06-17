#!/usr/bin/env python3
"""Update the class constructor def __init__(self, batch_size, max_len)"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """Loads and preps a dataset for machine translation"""
    def __init__(self, batch_size, max_len):
        """initialize"""
        self.batch_size = batch_size
        self.max_len = max_len

        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(self.filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(10000)
        self.data_train = self.data_train.padded_batch(self.batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(self.filter_max_len)
        self.data_valid = self.data_valid.padded_batch(self.batch_size, padded_shapes=([None], [None]))
    
    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset"""
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2**15)
        
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2**15)
        
        return tokenizer_pt, tokenizer_en
    
    def encode(self, pt, en):
        """Encodes a translation into token"""
        pt_start_token = [self.tokenizer_pt.vocab_size]
        pt_end_token = [self.tokenizer_pt.vocab_size + 1]
        en_start_token = [self.tokenizer_en.vocab_size]
        en_end_token = [self.tokenizer_en.vocab_size + 1]

        pt_tokens = pt_start_token + self.tokenizer_pt.encode(pt.numpy()) + pt_end_token
        en_tokens = en_start_token + self.tokenizer_en.encode(en.numpy()) + en_end_token

        return np.array(pt_tokens), np.array(en_tokens)
    
    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance method"""
        pt_tokens, en_tokens = tf.py_function(func=self.encode, inp=[pt, en], Tout=[tf.int64, tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        
        return pt_tokens, en_tokens
    
    def filter_max_len(self, pt, en):
        """Filters out sentences that have more than max_len tokens"""
        return tf.logical_and(tf.size(pt) <= self.max_len, tf.size(en) <= self.max_len)
