# -*- coding: utf-8 -*-
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = '0.0.1'

from evaluation import Evaluation
from preprocessing import *
from summary_model import SummaryModel
from eval_example_script import evaluate_examples
from word_mover_distance import WordMoverDistance
