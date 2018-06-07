
from config import config

import datetime
import json
from rouge import Rouge # https://github.com/pltrdy/rouge
"""
Performs all ROUGE tests. Outputs test reports to the rouge reports folder.
Filename: 'evaluation.py'
Methods:
    - test_all_articles(articles, type='recall')
    - test_article(article, type='recall')
"""

def test_all_articles(articles, type='recall'):
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

def test_article(article, type='recall'):
    """
    type can be 'recall', 'precision', or 'f-measure'
    """
    return test_all(
        hyps = [article.generated_summary for i in range(len(article.gold_summaries))],
        refs = article.gold_summaries,
        type = type
    )

def test_single(hypothesis, reference, type='recall'):
    """
    type can be 'recall', 'precision', or 'f-measure'
    """
    assert(type in ['recall','precision','f-measure'])
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)[0]
    scores = {
        'rouge-1':"{0:.2f}".format(scores['rouge-1'][type[0]] * 100),
        'rouge-2':"{0:.2f}".format(scores['rouge-2'][type[0]] * 100),
        'rouge-l':"{0:.2f}".format(scores['rouge-l'][type[0]] * 100),
        'test_type': type
    }
    if config['rouge']['output_report']:
        output_rouge_report(scores, type, 1)
    return scores

def test_all(hyps, refs, type='recall'):
    """
    type can be 'recall', 'precision', or 'f-measure'
    """
    assert(type in ['recall','precision','f-measure'])
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    scores = {
        'rouge-1':"{0:.2f}".format(scores['rouge-1'][type[0]] * 100),
        'rouge-2':"{0:.2f}".format(scores['rouge-2'][type[0]] * 100),
        'rouge-l':"{0:.2f}".format(scores['rouge-l'][type[0]] * 100),
        'test_type': type
    }
    if config['rouge']['output_report']:
        output_rouge_report(scores, type, len(set(hyps)))
    return scores

def output_rouge_report(results, test_type, num_tested):
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

def display_scores(scores):
    """
    Displays the ROUGE scores to the console.
    """
    print('----------------------')
    print('  ROUGE Score Report  ')
    print('----------------------')
    for k,v in scores.iteritems():
        print(str(k) + ": \t" + str(v))
    print('')
