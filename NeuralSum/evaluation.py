
from config import config

import datetime
import json
from rouge import Rouge # https://github.com/pltrdy/rouge
from InferSent import InferSent
from word_mover_distance import WordMoverDistance

"""
Performs all ROUGE tests and InferSent comparisons.
Outputs test reports to the rouge reports folder.
Filename: 'evaluation.py'
"""
class Evaluation(object):
    """
    Needs to be a class so that the InferSent data and WMD data
        is only loaded once. (time expensive)
    """
    def __init__(self):
        self.infersent = InferSent()
        self.wmd = WordMoverDistance()
        self.rouge = Rouge()

    def test_all_articles(self, articles, type='recall'):
        """
        type can be 'recall', 'precision', or 'f-measure'
        """
        hyps = []
        refs = []
        for article in articles:
            hyps += [article.generated_summary for i in range(len(article.gold_summaries))]
            refs += article.gold_summaries

        assert(len(hyps) == len(refs))
        return test_all(hyps, refs, type)

    def test_article(self, article, type='recall'):
        """
        type can be 'recall', 'precision', or 'f-measure'
        """
        return test_all(
            hyps = [article.generated_summary for i in range(len(article.gold_summaries))],
            refs = article.gold_summaries,
            type = type
        )

    def test_single(self, hypothesis, reference, type='recall'):
        """
        type can be 'recall', 'precision', or 'f-measure'
        """
        assert(type in ['recall','precision','f-measure'])

        scores = self.rouge.get_scores(hypothesis, reference)[0]
        scores = {
            'rouge-1':"{0:.2f}".format(scores['rouge-1'][type[0]] * 100),
            'rouge-2':"{0:.2f}".format(scores['rouge-2'][type[0]] * 100),
            'rouge-l':"{0:.2f}".format(scores['rouge-l'][type[0]] * 100),
            'cos_sim':str(self.infersent.get_avg_similarity([hypothesis], [reference])),
            'word_mover_distance':self.wmd.get_wmd(hypothesis, reference),
            'test_type': type
        }
        if config['rouge']['output_report']:
            self.output_eval_report(scores, type, 1)
        return scores

    def test_all(self, hyps, refs, type='recall'):
        """
        type can be 'recall', 'precision', or 'f-measure'
        """
        assert(type in ['recall','precision','f-measure'])
        scores = self.rouge.get_scores(hyps, refs, avg=True)
        scores = {
            'rouge-1':"{0:.2f}".format(scores['rouge-1'][type[0]] * 100),
            'rouge-2':"{0:.2f}".format(scores['rouge-2'][type[0]] * 100),
            'rouge-l':"{0:.2f}".format(scores['rouge-l'][type[0]] * 100),
            'cos_sim':str(self.infersent.get_avg_similarity(hyps, refs)),
            'word_mover_distance':self.wmd.get_avg_wmd(hyps, refs),
            'test_type': type
        }
        if config['rouge']['output_report']:
            self.output_eval_report(scores, type, len(set(hyps)))
        return scores

    def output_eval_report(self, results, test_type, num_tested):
        """
        Outputs the given results dictionary to a json file in
            the rouge reports folder
        Args:
            results (dict): rouge results
            test_type (string): type of rouge test. ex: 'recall'
            num_tested (int): number of unique hypothesis summaries
        Returns:
            None
        """
        dt = str(datetime.datetime.now().isoformat())[:-5]
        name = "rouge_" + test_type + "_" + dt
        filename = config["rouge"]["reports_folder"] + name + ".json"
        with open(filename, 'w') as f:
            results["num_tested"] = num_tested
            json.dump(results, f, indent=2, sort_keys=True)

    def display_scores(self, scores):
        """
        Displays the ROUGE scores to the console.
        """
        print('----------------------')
        print('  ROUGE Score Report  ')
        print('----------------------')
        for k,v in scores.iteritems():
            print(str(k) + ": \t" + str(v))
        print('')
