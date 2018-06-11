
from InferSent import InferSent
import NeuralSum as ns

import unittest

class TestInferSent(unittest.TestCase):

    def setUp(self):
        self.infersent = InferSent(
            ns.get_vocabulary(ns.parse_duc_2004() + ns.parse_duc_2003())
        )
        self.sentences = [
            'Peter would like his dog to jump over the fence.',
            'Peter does not like his dog jumping over the fence.',
            'Peter loves when his dog jumps over the fence.',
            'this is a sample sentence to test.',
            'so is this one, but it slightly different.',
            'All things are better in groups of threes'
        ]
        self.embeddings = None

    def tearDown(self):
        self.infersent = None
        self.sentences = None

    def test_get_vectors(self):
        if self.embeddings is not None:
            return

        self.embeddings = self.infersent.get_embeddings(self.sentences)
        self.assertEqual(len(self.embeddings), len(self.sentences))

        # dimension of each vector is supposed to be 4096:
        self.assertEqual(len(self.embeddings[0]), 4096)

        # code for visualizing word importance within sentences:
        # infersent.visualize('this is a sample sentence to test.')

    def test_cosine_similarity(self):
        if self.embeddings is None:
            self.test_get_vectors()

        # use .reshape(1, -1) because we have a single sample [ ... ]
        sim1 = self.infersent.cosine_similarity(
            self.embeddings[0].reshape(1, -1),
            self.embeddings[1].reshape(1, -1)
        )
        sim2 = self.infersent.cosine_similarity(
            self.embeddings[0].reshape(1, -1),
            self.embeddings[2].reshape(1, -1)
        )

        # expect the second sentence to be semantically closer to
        #   the original
        self.assertGreater(sim2[0][0], sim1[0][0])

        self.assertGreater(0.50, self.infersent.cosine_similarity(
            self.embeddings[0], self.embeddings[-1])[0][0]
        )

    def test_avg_similarity(self):
        avg = self.infersent.get_avg_similarity(self.sentences[:3], self.sentences[3:])
        print avg
