import NeuralSum as ns

import unittest

class TestWordMoverDistance(unittest.TestCase):

    def setUp(self):
        self.wmd = ns.WordMoverDistance()
        self.debug = False

    def tearDown(self):
        del self.wmd
        del self.debug

    def test_wmd(self):
        sen1 = 'Obama speaks to the media in Illinois'
        sen2 = 'The president greets the press in Chicago'
        wmd1 = self.wmd.get_wmd(sen1, sen2)

        sen1 = 'The president greets the press in Chicago'
        sen2 = 'Obama speaks in Illinois'
        wmd2 = self.wmd.get_wmd(sen1, sen2)

        self.assertGreater(wmd2, wmd1)

        self.AssertEqual(self.wmd.get_wmd(
            'beef and veggies are tasty',
            'beef and veggies are tasty',
        ), 0.0)

        if not self.debug:
            return


        print('WMD1: ' + str(wmd1))
        print('WMD2: ' + str(wmd2))

        sen1 = 'beef and veggies are tasty'
        sen2 = 'beef and veggies are tasty'
        wmd = self.wmd.get_wmd(sen1, sen2)
        print('WMD3: ' + str(wmd))

        sen1 = 'beef and veggies are tasty'
        sen2 = 'beef and veggies are delicious'
        wmd = self.wmd.get_wmd(sen1, sen2)
        print('WMD4: ' + str(wmd))

        sen1 = 'beef and veggies are tasty'
        sen2 = 'jugular appendages dancing vigorously'
        wmd = self.wmd.get_wmd(sen1, sen2)
        print('WMD5: ' + str(wmd))

        sen1 = 'fudge'
        sen2 = 'jugular appendages dancing vigorously upon the cozy hearth at winters night'
        wmd = self.wmd.get_wmd(sen1, sen2)
        print('WMD6: ' + str(wmd))

        gens = [
            'hundreds of a voice in state state newspapers newspapers newspapers newspapers and jailed jailed opposition leader leader have turned to the internet to the internet',
            'indonesian president clinton of ``finds finds finds a summit of asia-pacific ``difficult because of ``asked his concerns about the arrest of malaysias former deputy deputy president',
            'among among asias leaders prime minister mahathir mohamad was mohamad as a man a bold: a physical and social social social social social that this party into the world affairs',
            'on on on the face of dissident anwar ibrahim on newspaper front front river for two days and from the the internet are are are unconstitutional and hundreds of malaysias'
        ]
        sums = [
            'anwar supporters speak out on internet unblocked by government',
            'regional leaders consider boycotting malaysian meeting due to anwar arrest',
            'mahathirs 17 years saw great advances now economic crisis instability',
            'malaysian prime minister expresses surprise at behavior of his police'
        ]

        print('')
        for i in range(len(gens)):
            print(gens[i])
            print(sums[i])
            print(self.wmd.get_wmd(gens[i], sums[i]))
