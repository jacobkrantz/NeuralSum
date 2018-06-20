
from config import config

from itertools import starmap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

class InferSent(object):
    """Interacts with the pretrained Infersent module from Facebook."""
    def __init__(self, vocab_size=None):
        """
        IMPORTANT: keep the vocab very large: the more words in the vocab, the
            more differentiation between sentences. This gives better semantic
            similarities. Downside is the larger the vocab, the longer the init
            process takes.
            Config can specify vocab size up to 2.2 million.
        Args:
            vocab_size (int): size of vocab to load into the word vector space
                for determining sentence vector similarity. Optional.
        """
        self.infersent = infersent = torch.load(
            config['infersent']['trained_model'],
            map_location=lambda storage,
            loc: storage
        )
        self.infersent.set_glove_path(config['infersent']['glove_file'])
        if vocab_size is None:
            vocab_size = config['infersent']['vocab_size']
        self.infersent.build_vocab_k_words(vocab_size)

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
        """
        Returns:
            (float): the mean similarity,
            (list<float>): list of calculated similarities
        """
        assert(len(sentences) == len(summaries))
        embeddings_1 = self.get_embeddings(sentences)
        print('InferSent: Embeddings generated for group 1.')
        embeddings_2 = self.get_embeddings(summaries)
        print('InferSent: Embeddings generated for group 2.')

        sims = list(starmap(
            lambda e1,e2: self.cosine_similarity(e1,e2),
            zip(embeddings_1, embeddings_2)
        ))
        return np.mean(sims), sims

    def cosine_similarity(self, sentence_1, sentence_2):
        """
        Only call with one sentence at a time.
        """
        return cosine_similarity(
            sentence_1.reshape(1,-1),
            sentence_2.reshape(1,-1)
        )[0][0]
