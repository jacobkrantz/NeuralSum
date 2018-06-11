
from config import config

import nltk
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

    def get_vectors(self, sentences):
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
