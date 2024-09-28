#!/usr/bin/env python3
""" Bag of Words """


import numpy as np


def bag_of_words(sentences, vocab=None):
    '''
    Creates a bag of words embedding matrix

    Args:
        - sentences: a list of sentences to analyze
        - vocab: a list of vocabulary words to use

    Returns:
        - embeddings: np arrag of shape (s, f)
            - s is the number of sentences
            - f is the number of features analyzed
        - features: list of features used for embeddings
    '''


if __name__ == '__main__':
    sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]

    E, F = bag_of_words(sentences=sentences)
    print(E)
    print(F)
