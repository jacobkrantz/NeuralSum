
from config import config

from itertools import starmap
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import download
from nltk.downloader import Downloader
import numpy as np
import os

class WordMoverDistance(object):
    """
    Computes the word mover distance between two sentences.
    A measure of dissimilarity.
    Vectors sourced from pretrained Word2Vec.
    Note that the values for WMD differ from those in the original paper.

    Sources:
        https://radimrehurek.com/gensim/models/keyedvectors.html
        GenSim: free, statistical semantics python package

        @inproceedings{rehurek_lrec,
              title = {{Software Framework for Topic Modelling with Large Corpora}},
              author = {Radim {\v R}eh{\r u}{\v r}ek and Petr Sojka},
              booktitle = {{Proceedings of the LREC 2010 Workshop on New
                   Challenges for NLP Frameworks}},
              pages = {45--50},
              year = 2010,
              month = May,
              day = 22,
              publisher = {ELRA},
              address = {Valletta, Malta},
              note={\url{http://is.muni.cz/publication/884893/en}},
              language={English}
        }
    """
    def __init__(self):
        # check for stopwords installation
        if not Downloader().is_installed('stopwords'):
            download('stopwords')

        if config['wmd']['save_memory']:
            # 25% slower but 50% memory savings by reducing datatype size.
            self.model = KeyedVectors.load_word2vec_format(
                config['wmd']['word2vec'],
                binary=True,
                datatype=np.float16
            )
        else:
            self.model = KeyedVectors.load_word2vec_format(
                config['wmd']['word2vec'],
                binary=True
            )
        if config['wmd']['normalize']:
            # computes L2-norms of word weight vectors
            self.model.init_sims(replace=True)

        self.stopwords = stopwords.words('english')

    def get_avg_wmd(self, sentences_1, sentences_2):
        """
        Calculates the average word mover's distance element-wise between
            two lists of sentences.
        Returns:
            (float): the mean distance,
            (list<float>): list of calculated distances
        """
        assert(len(sentences_1) == len(sentences_2))

        distances = list(starmap(
            lambda s1,s2: self.get_wmd(s1,s2),
            zip(sentences_1, sentences_2)
        ))
        return np.mean(distances), distances

    def get_wmd(self, sentence_1, sentence_2):
        """
        Calculates the word mover's distance for a single sentence pair.
        If the distance is greater than 5.0, truncates to 5.0. This is because
        model.wmdistance() occasionally returns 'inf'.
        Returns:
            float: WMD with a range of [0.0,5.0]
        """
        sentence_1 = sentence_1.lower().split()
        sentence_2 = sentence_2.lower().split()

        sentence_1 = [w for w in sentence_1 if w not in self.stopwords]
        sentence_2 = [w for w in sentence_2 if w not in self.stopwords]

        max_dist = 5.0
        dist = self.model.wmdistance(sentence_1, sentence_2)
        return dist if dist < max_dist else max_dist
