#!/usr/bin/env python3
""" finds a snippet of text within a
reference document to answer a question"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Load the pre-trained BERT model and tokenizer"""
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")

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

    start_index = tf.argmax(start_logits).numpy()
    end_index = tf.argmax(end_logits).numpy()

    if start_index <= end_index:
        answer_tokens = tokens[start_index:end_index + 1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
    else:
        answer = None

    return answer
