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
        wmd1 = self.wmd.get_wmd(sen1, sen2)

        sen1 = 'The president greets the press in Chicago'
        sen2 = 'Obama speaks in Illinois'
        wmd2 = self.wmd.get_wmd(sen1, sen2)

        print('WMD1: ' + str(wmd1))
        print('WMD2: ' + str(wmd2))
        self.assertGreater(wmd2, wmd1)

        self.AssertEqual(self.wmd.get_wmd(
            'beef and veggies are tasty',
            'beef and veggies are tasty',
        ), 0.0)

