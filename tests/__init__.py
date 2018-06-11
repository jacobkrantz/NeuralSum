import unittest
from .test_infersent import TestInferSent
from .test_neural_sum import TestNeuralSum

def test_suite():
    loader = unittest.TestLoader()

    test_classes_to_run = [TestInferSent, TestNeuralSum]
    suites_list = []

    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    return unittest.TestSuite(suites_list)
