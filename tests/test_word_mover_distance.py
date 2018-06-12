import NeuralSum as ns

import unittest

class TestWordMoverDistance(unittest.TestCase):

    def setUp(self):
        self.wmd = ns.WordMoverDistance()

    def tearDown(self):
        self.wmd = None

    def test_wmd(self):
        sen1 = 'Obama speaks to the media in Illinois'
        sen2 = 'The president greets the press in Chicago'
        print self.wmd.get_wmd(sen1, sen2)

        sen1 = 'The president greets the press in Chicago'
        sen2 = 'Obama speaks in Illinois'
        print self.wmd.get_wmd(sen1, sen2)
