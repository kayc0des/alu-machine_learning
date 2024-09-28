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

    # Text preprocessing -> lower sentences
    lowered = []
    for sentence in sentences:
        lowered.append(sentence.lower())

    # Remove exclamation marks and 's
    lowered = [sentence.replace('!', '') for sentence in lowered]
    lowered = [sentence.replace("'s", '') for sentence in lowered]

    # Create dictionary
    if vocab is None:
        unique_words = set()
        for sentence in lowered:
            for word in sentence.split():
                unique_words.add(word)
        vocab = list(unique_words)
        vocab.sort(reverse=False)

    f = len(vocab)
    s = len(lowered)

    # initialize BOW embeddings
    embeddings = np.zeros(shape=(s, f), dtype=np.int32)

    for i in range(s):
        sentence = lowered[i]
        words = sentence.split()
        for j in range(f):
            for word in words:
                if word in vocab:
                    freq = words.count(word)
                    index = vocab.index(word)
                    embeddings[i][index] = freq

    return embeddings, vocab


if __name__ == '__main__':
    sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]

    E, F = bag_of_words(sentences)
    print(E)
    print(F)
    print(E.shape)
