#!/usr/bin/env python3
"""updates class Dataset: encodes a translation into tokens"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """Loads and preps a dataset for machine translation"""
    def __init__(self):
        """initialize"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)
    
    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset"""
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2**15)
        
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2**15)
        
        return tokenizer_pt, tokenizer_en
    
    def encode(self, pt, en):
        """Encodes a translation into tokens"""
        pt_start_token = [self.tokenizer_pt.vocab_size]
        pt_end_token = [self.tokenizer_pt.vocab_size + 1]
        en_start_token = [self.tokenizer_en.vocab_size]
        en_end_token = [self.tokenizer_en.vocab_size + 1]

        pt_tokens = pt_start_token + self.tokenizer_pt.encode(pt.numpy()) + pt_end_token
        en_tokens = en_start_token + self.tokenizer_en.encode(en.numpy()) + en_end_token

        return np.array(pt_tokens), np.array(en_tokens)
    