from __future__ import print_function

from NeuralSum import Vert
from config import config

import fire
import logging as log

"""
script for evaluating the tensor2tensor generated summaries that are located
    in the output location specified in the decoding shell script.
- opens up generated.txt and summaries.txt
- compares line by line using the evaluation tool in NeuralSum.
- outputs score report to file as specified in config or command line argument.
- uses about 7GB of memory. Could be reduced by half, but major rework
    of NeuralSum.Evaluation would have to be done.
Usage:
    >>> python evaluate_on_duc.py [--which_duc=(2003|2004)] [--save_report=(True|False)]
Requirements:
    generated.txt and summaries.txt in respective DUC folder.
"""

def evaluate(which_duc='2003', save_report=True):
    """
    Args:
        which_duc (string): either 2003 or 2004. Default: 2003.
        save_report (bool): if True, saves report to location specified in
            the config. Default: True.
    Returns:
        dict of scores
    """
    generated_f = './data/' + which_duc + '/generated.txt'
    summaries_f = './data/' + which_duc + '/summaries.txt'

    log.info("Starting step: read data files.")
    generated = list()

    try:
        with open(generated_f, mode='r') as gen_f:
            for line in gen_f:
                generated.append(line.strip('\n'))
    except IOError as e:
        raise IOError(str(e) + '\n' +
            'Try running `python process_duc.py <duc_year>` to load the data')

    summaries = list()
    try:
        with open(summaries_f, mode='r') as summ_f:
            for line in summ_f:
                summaries.append(line.strip('\n'))
    except IOError as e:
        print(e)
        print('Try running `python process_duc.py <duc_year>` to load the data')

    assert(len(generated) == len(summaries))
    log.info("Finished step: read data files.")

    log.info("Starting step: Evaluating " + str(len(generated)) + " summary pairs.")
    vert = Vert()
    scores = vert.score(
        generated,
        summaries,
        rouge_type=config['vert']['rouge_metric'],
        verbose=True
    )
    vert.display_scores(scores)
    if save_report:
        vert.output_report(scores, config['vert']['reports_folder'])
    log.info("Finished step: Evaluate summary pairs.")
    return scores

if __name__ == '__main__':
    log.getLogger()
    fire.Fire(evaluate)
