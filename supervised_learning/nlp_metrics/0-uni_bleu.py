#!/usr/bin/env python3
"""calculates the unigram BLEU score for a sentence"""

import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence"""
    len_sentence = len(sentence)

    ref_lengths = [len(ref) for ref in references]

    closest_ref_len = min(ref_lengths, key=lambda ref_len: (abs(ref_len - len_sentence), ref_len))

    sentence_unigrams = set(sentence)
    max_counts = {}
    for word in sentence_unigrams:
        max_counts[word] = max(ref.count(word) for ref in references)

    clipped_count = sum(min(sentence.count(word), max_counts[word]) for word in sentence_unigrams)
    precision = clipped_count / len_sentence

    if len_sentence > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / len_sentence)

    bleu_score = brevity_penalty * precision
    return bleu_score
