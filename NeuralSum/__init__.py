# -*- coding: utf-8 -*-
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = '0.0.1'

from preprocessing import *
from summary_model import SummaryModel
from word_mover_distance import WordMoverDistance
from vert import Vert

# for tensor2tensor
from . import summary_problems
from . import summary_modalities
from . import my_custom_hparams
from . import my_custom_transformer
