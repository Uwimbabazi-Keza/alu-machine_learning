#!/usr/bin/env python3
"""performs semantic search on a corpus of documents"""

from sentence_transformers import SentenceTransformer, util
import os

def semantic_search(corpus_path, sentence):
    """Load pre-trained sentence transformer model"""
    model = SentenceTransformer('all-mpnet-base-v2')

    corpus = []
    file_names = []
    for file_name in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            corpus.append(file.read())
            file_names.append(file_name)
    
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(sentence, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)

    best_match_idx = int(cosine_scores.argmax())
    best_match_text = corpus[best_match_idx]
    best_match_file = file_names[best_match_idx]

    return best_match_file, best_match_text
