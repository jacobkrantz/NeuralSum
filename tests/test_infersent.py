
from InferSent import InferSent
import NeuralSum as ns

import unittest

class TestInferSent(unittest.TestCase):

    def setUp(self):
        self.infersent = InferSent(
            ns.get_vocabulary(ns.parse_duc_2004() + ns.parse_duc_2003())
        )
        self.sentences = [
            'this is a sample sentence to test.',
            'so is this one, but it slightly different.',
            'All things are better in groups of threes'
        ]

    def tearDown(self):
        self.infersent = None
        self.sentences = None

    def test_get_vectors(self):
        vectors = self.infersent.get_vectors(self.sentences)
        self.assertEqual(len(vectors), len(self.sentences))

        # dimension of each vector is supposed to be 4096:
        self.assertEqual(len(vectors[0]), 4096)

        # code for visualizing word importance within sentences:
        # infersent.visualize('this is a sample sentence to test.')
