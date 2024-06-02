#!/usr/bin/env python3
"""creates a bag of words embedding matrix"""

import numpy as np
from collections import defaultdict


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix"""
    tokenized_sentences = [sentence.split() for sentence in sentences]

    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    
    word_to_index = {word: i for i, word in enumerate(vocab)}

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, sentence in enumerate(tokenized_sentences):
        word_counts = defaultdict(int)
        for word in sentence:
            if word in word_to_index:
                word_counts[word] += 1
        for word, count in word_counts.items():
            embeddings[i, word_to_index[word]] = count

    features = vocab
    
    return embeddings, features
