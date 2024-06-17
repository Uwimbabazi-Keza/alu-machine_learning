#!/usr/bin/env python3
"""calculates the n-gram BLEU score for a sentence"""

import numpy as np
from collections import Counter


def compute_ngrams(sentence, n):
    """Compute n-grams for a given sentenc"""
    return [tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1)]

def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence"""
    len_sentence = len(sentence)
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lengths, key=lambda ref_len: (abs(ref_len - len_sentence), ref_len))
    sentence_ngrams = compute_ngrams(sentence, n)
    max_counts = {}
    for ref in references:
        ref_ngrams = compute_ngrams(ref, n)
        ref_ngram_counts = Counter(ref_ngrams)
        for ngram in ref_ngram_counts:
            max_counts[ngram] = max(
                max_counts.get(ngram, 0), ref_ngram_counts[ngram])
    
    clipped_count = sum(min
                        (sentence_ngrams.count(ngram), max_counts.get(ngram, 0)
                         ) for ngram in set(sentence_ngrams))
    precision = clipped_count / max(1, len(sentence_ngrams))

    if len_sentence > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / len_sentence)

    bleu_score = brevity_penalty * precision
    return bleu_score
