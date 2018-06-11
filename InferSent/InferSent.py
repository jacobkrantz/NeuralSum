
from config import config

from itertools import starmap
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

class InferSent(object):
    """Interacts with the pretrained Infersent module from Facebook."""
    def __init__(self, vocab):
        """
        Args:
            vocab (list<string>) list of sentences to be included in the vocab.
        """
        self.infersent = infersent = torch.load(
            config['infersent']['trained_model'],
            map_location=lambda storage,
            loc: storage
        )
        self.infersent.set_glove_path(config['infersent']['glove_file'])
        self.infersent.build_vocab(vocab, tokenize=True)

    def get_embeddings(self, sentences):
        """
        Args:
            sentences (list<string>) list of sentences
        Returns:
            numpy array of embeddings of dim=4096
        """
        embeddings = self.infersent.encode(sentences, tokenize=True)
        return embeddings

    def visualize(self, sentence):
        self.infersent.visualize(sentence, tokenize=True)

    def get_avg_similarity(self, sentences, summaries):
        assert(len(sentences) == len(summaries))

        embeddings_1 = self.get_embeddings(sentences)
        embeddings_2 = self.get_embeddings(summaries)

        return np.mean(list(starmap(
            lambda e1,e2: self.cosine_similarity(e1,e2),
            zip(embeddings_1, embeddings_2)
        )))


    def cosine_similarity(self, sentence_1, sentence_2):
        """
        Only call with one sentence at a time.
        """

        return cosine_similarity(
            sentence_1.reshape(1,-1),
            sentence_2.reshape(1,-1)
        )
