#!/usr/bin/env python3
""" Term frequency - inverse document frequency """


from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    '''
    Creates a tf-idf embedding matrix

    Args:
        - sentences: a list of sentences to analyze
        - vocab: a list of vocabulary words to use

    Returns:
        - embeddings: np arrag of shape (s, f)
            - s is the number of sentences
            - f is the number of features analyzed
        - features: list of features used for embeddings
    '''

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embeddings = x.toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
