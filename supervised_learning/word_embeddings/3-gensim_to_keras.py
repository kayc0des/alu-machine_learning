#!/usr/bin/env python3
""" COnvers a gensim model to keras """

def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to keras
    embedding layer

    Args:
        model: gensim word2vec model

    Returns:
        keras embedding layer
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
