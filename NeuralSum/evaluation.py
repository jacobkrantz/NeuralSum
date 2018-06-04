
from config import config

import datetime
import json
from rouge import Rouge # https://github.com/pltrdy/rouge
"""
filename: evaluation.py
- file for performing all ROUGE tests. Outputs test reports
    to the rouge reports folder
"""

def test_single(hypothesis, reference, type='recall'):
    """
    type can be 'recall', 'precision', or 'f-measure'
    """
    assert(type in ['recall','precision','f-measure'])
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)[0]
    scores = {
        'rouge-1':scores['rouge-1'][type[0]],
        'rouge-2':scores['rouge-2'][type[0]],
        'rouge-l':scores['rouge-l'][type[0]]
    }
    if config['rouge']['output_report']:
        output_rouge_report(scores, type)
    return scores

def test_all(hyps, refs, type='recall'):
    """
    type can be 'recall', 'precision', or 'f-measure'
    """
    assert(type in ['recall','precision','f-measure'])
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    if config['rouge']['output_report']:
        output_rouge_report(scores, type)
    return scores

def output_rouge_report(results, test_type):
    """
    Outputs the given results dictionary to a json file in
        the rouge reports folder
    Args:
        results (dict): rouge results
        test_type (string): type of rouge test. ex: 'recall'
    Returns:
        None
    """
    dt = str(datetime.datetime.now().isoformat())[:-5]
    name = "rouge_" + test_type + "_" + dt
    filename = config["rouge"]["reports_folder"] + name + ".json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
