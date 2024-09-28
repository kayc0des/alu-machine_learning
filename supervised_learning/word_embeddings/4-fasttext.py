from gensim.models import FastText
from gensim.utils import simple_preprocess

def fasttext_model(sentences,
                   size=100,
                   min_count=5,
                   negative=5,
                   window=5,
                   cbow=True,
                   iterations=5,
                   seed=0,
                   workers=1):
    '''
    Creates and trains a Gensim FastText model.

    Args:
        - sentences: a list of sentences to be trained on (list of strings)
        - size: the dimensionality of the embedding layer
        - min_count: the minimum number of occurrences of a word for use in training
        - window: the maximum distance between the current and predicted word within a sentence
        - negative: the size of negative sampling
        - cbow: a boolean to determine the training type (True = CBOW, False = Skip-gram)
        - iterations: the number of iterations to train over
        - seed: the seed for the random number generator
        - workers: the number of worker threads to train the model

    Returns:
        - The trained FastText model
    '''

    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
    
    # Create the FastText model
    model = FastText(
        sentences=tokenized_sentences,
        vector_size=size,
        window=window,
        min_count=min_count,
        sg=0 if cbow else 1,
        negative=negative,
        seed=seed,
        workers=workers
    )

    # Train the model
    model.train(
        tokenized_sentences,
        total_examples=len(tokenized_sentences),
        epochs=iterations
    )

    return model
