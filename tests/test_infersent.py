
from InferSent import InferSent
import NeuralSum as ns

import unittest

class TestInferSent(unittest.TestCase):

    def setUp(self):
        self.infersent = InferSent(vocab_size=500000)
        self.sentences = [
            'Peter would like his dog to jump over the fence.',
            'Peter does not like his dog jumping over the fence.',
            'Peter loves when his dog jumps over the fence.',
            'this is a sample sentence to test.',
            'so is this one, but it slightly different.',
            'All things are better in groups of threes',
            'there are specific language governing permissions and limitations under the license',
            'buns'
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
        self.assertGreater(sim2, sim1)

        self.assertGreater(0.50, self.infersent.cosine_similarity(
            self.embeddings[0], self.embeddings[-1])
        )

        # expected to have the same meaning vector
        self.assertAlmostEqual(1.0, self.infersent.cosine_similarity(
            self.embeddings[0].reshape(1, -1),
            self.embeddings[0].reshape(1, -1)
        ))

        # expected to be very different
        self.assertGreater(0.50, self.infersent.cosine_similarity(
            self.embeddings[-1].reshape(1, -1),
            self.embeddings[-2].reshape(1, -1)
        ))

    def test_avg_similarity(self):
        avg = self.infersent.get_avg_similarity(self.sentences[:3], self.sentences[3:6])
        self.assertGreater(avg, 0.0)
        self.assertGreater(1.0, avg)
