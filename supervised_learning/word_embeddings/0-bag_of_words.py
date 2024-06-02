#!/usr/bin/env python3
"""creates a bag of words embedding matrix"""

import numpy as np


def bag_of_words(sentences, vocab=None):
    tokenized_sentences = [sentence.split() for sentence in sentences]
    
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1
    
    return embeddings, vocab
