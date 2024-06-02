#!/usr/bin/env python3
"""creates a TF-IDF embedding"""

import numpy as np
from collections import Counter
from math import log


def tf_idf(sentences, vocab=None):
    tokenized_sentences = [sentence.split() for sentence in sentences]
    
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    tf_matrix = np.zeros((len(sentences), len(vocab)), dtype=float)
    
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)
        for word, count in word_counts.items():
            if word in word_to_index:
                tf_matrix[i, word_to_index[word]] = count / len(sentence)
    
    df = np.zeros(len(vocab), dtype=float)
    for i, word in enumerate(vocab):
        df[i] = sum(word in sentence for sentence in tokenized_sentences)
    
    idf = np.log(len(sentences) / (df + 1))  # Adding 1 to avoid division by zero
    
    tf_idf_matrix = tf_matrix * idf
    
    return tf_idf_matrix, vocab
