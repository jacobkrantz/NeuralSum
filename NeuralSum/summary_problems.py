
from preprocessing import *

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

"""
Problems:
- summary_problem
    - uses all but last 10k articles of Gigaword.
- summary_problem_small
    - uses 25% of Gigaword.
"""

@registry.register_problem
class SummaryProblem(text_problems.Text2TextProblem):
    """
    Use my own data for summarization.
    To run in the terminal:
        modify datagen.sh as needed. Run from ~Code/NeuralSum/ as such:
        >>> ./datagen.sh
    """
    @property
    def approx_vocab_size(self):
        return 2**14  # ~16k

    @property
    def is_generate_per_split(self):
        """ generate_data will shard the data into TRAIN and EVAL for us. """
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def vocab_type(self):
        """What kind of vocabulary to use.
        `VocabType`s:
          * `SUBWORD`: `SubwordTextEncoder`, an invertible wordpiece vocabulary.
            Must provide `self.approx_vocab_size`. Generates the vocabulary based on
            the training data. To limit the number of samples the vocab generation
            looks at, override `self.max_samples_for_vocab`. Recommended and
            default.
          * `CHARACTER`: `ByteTextEncoder`, encode raw bytes.
          * `TOKEN`: `TokenTextEncoder`, vocabulary based on a file. Must provide a
            vocabulary file yourself (`TokenTextEncoder.store_to_file`) because one
            will not be generated for you. The vocab file should be stored in
            `data_dir/` with the name specified by `self.vocab_filename`.
        Returns:
          VocabType constant
        """
        return text_problems.VocabType.SUBWORD

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """
        Steps:
        1. Extract raw files from tmp_dir
        2. Tokenize/process data
        3.  loop over all sentence-summary pairs and yield them

        We use yield instead of return to return a generator that can only
            be iterated over once. Saves on memory.
        """
        del data_dir, tmp_dir, dataset_split

        # leaves out 10k articles for eval if desired in the future.
        articles = parse_gigaword(limit=3793957, randomize=False)

        for art in articles:
            for summ in art.gold_summaries:
                yield {
                    "inputs": art.sentence,
                    "targets": summ
                }

    def eval_metrics(self):
        """ Override: ignore some metrics """
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.NEG_LOG_PERPLEXITY,
            metrics.Metrics.ROUGE_2_F,
            metrics.Metrics.ROUGE_L_F
        ]

@registry.register_problem
class SummaryProblemSmall(text_problems.Text2TextProblem):
    """
    Use my own data for summarization. Limit Gigaword to 25%.
    To run in the terminal:
        modify datagen.sh as needed. Run from ~Code/NeuralSum/ as such:
        >>> ./datagen.sh
    """
    @property
    def approx_vocab_size(self):
        # made this larger just in case.
        return 2**15  # ~32k

    @property
    def is_generate_per_split(self):
        """ generate_data will shard the data into TRAIN and EVAL for us. """
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """
        Steps:
        1. Extract raw files from tmp_dir
        2. Tokenize/process data
        3.  loop over all sentence-summary pairs and yield them

        We use yield instead of return to return a generator that can only
            be iterated over once. Saves on memory.
        """
        del data_dir, tmp_dir, dataset_split
        articles = parse_gigaword(limit=950989, randomize=False) # which leaves out 10k articles.

        for art in articles:
            for summ in art.gold_summaries:
                yield {
                    "inputs": art.sentence,
                    "targets": summ
                }

    def eval_metrics(self):
        """ Override: ignore some metrics """
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.NEG_LOG_PERPLEXITY,
            metrics.Metrics.ROUGE_2_F,
            metrics.Metrics.ROUGE_L_F
        ]
