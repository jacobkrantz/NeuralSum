from abc import ABCMeta, abstractmethod

class AbstractModel(object):
    """
    An abstract class detailing all methods that must be present in our
        machine learning models.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def compile():
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def test_single(self):
        pass

    @abstractmethod
    def load_weights(self):
        pass
