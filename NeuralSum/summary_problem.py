
from preprocessing import *

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class SummaryProblem(text_problems.Text2TextProblem):
    """
    Use my own data for summarization.
    Run one directory up from here.
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

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """
        Steps:
        1. Extract raw files from tmp_dir
        2. Tokenize/process data
        3.  loop over all sentence-summary pairs and yield them

        We use yield instead of return to return a generator that can only
            be iterated over once. Saves on memory.
        """
        del data_dir
        del tmp_dir
        del dataset_split

        duc_2003_articles = parse_duc_2003()
        duc_2004_articles = parse_duc_2004()
        articles = duc_2003_articles + duc_2004_articles

        num_to_check = 5
        num_checked = 0
        print('Data entry sample should be shown below:')
        for art in articles:
            for summ in art.gold_summaries:
                if num_checked < num_to_check:
                    print {"inputs": art.sentence, "targets": summ}
                    num_checked += 1

                yield {
                    "inputs": art.sentence,
                    "targets": summ
                }
