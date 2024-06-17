#!/usr/bin/env python3
"""answers questions from multiple reference texts"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os


def get_answer(question, reference):
    """Define the question_answer function to get the answer from a single reference document"""
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)
    
    tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + reference_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(question_tokens) + 2) + [1] * (len(reference_tokens) + 1)
    
    input_ids = tf.constant(input_ids)[None, :]
    input_mask = tf.constant(input_mask)[None, :]
    segment_ids = tf.constant(segment_ids)[None, :]
    
    outputs = model([input_ids, input_mask, segment_ids])
    
    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()
    
    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)
    
    if start_index <= end_index and end_index < len(tokens):
        answer_tokens = tokens[start_index:end_index + 1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
    else:
        answer = None
    
    return answer

def semantic_search(corpus_path, sentence):
    """Define the semantic_search function to find the most relevant document"""    
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

    return best_match_text

def question_answer(corpus_path):
    """Define the question_answer function that answers questions from multiple reference texts"""
    while True:
        user_input = input("Q: ").strip()
        if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        else:
            best_reference = semantic_search(corpus_path, user_input)
            
            answer = get_answer(user_input, best_reference)
            if answer:
                print(f"A: {answer}")
            else:
                print("A: Sorry, I do not understand your question.")
