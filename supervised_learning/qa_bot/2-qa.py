#!/usr/bin/env python3
"""answers questions from a reference textt"""

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def answer_loop(reference):
    """Tokenize the reference text into sentences"""
    sentences = sent_tokenize(reference)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    
    def find_best_match(question):
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X)
        idx = similarities.argmax()
        return sentences[idx], similarities[0, idx]
    
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']
    print("Welcome! Ask me a question or type 'exit' to end.")
    
    while True:
        user_input = input('Q: ').strip().lower()
        
        if user_input in exit_commands:
            print('A: Goodbye')
            break
        
        try:
            best_match, similarity = find_best_match(user_input)
            if similarity < 0.2:
                raise ValueError("No suitable answer found")
            
            print(f'A: {best_match}')
        
        except:
            print("A: Sorry, I do not understand your question.")

