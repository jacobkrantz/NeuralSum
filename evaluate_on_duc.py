from __future__ import print_function

from NeuralSum import Vert
from config import config

import logging as log
import sys
"""
script for evaluated the tensor2tensor generated summaries that are located
    in the output location specified in the decoding shell script.
- opens up generated.txt and summaries.txt
- compares line by line using the evaluation tool in NeuralSum.
- outputs score report to file as specified in config or command line argument.
- uses about 7GB of memory. Could be reduced by half, but major rework
    of NeuralSum.Evaluation would have to be done.
To run:
>>> python evaluate_on_duc.py
To override whether or not a score report is saved:
>>> python evaluate_on_duc.py save
>>> python evaluate_on_duc.py toss
"""

GENERATED = './data/duc2004/generated.txt'
SUMMARIES = './data/duc2004/summaries.txt'

def main(save_scores):
    log.info("Starting step: read data files.")
    generated = list()
    with open(GENERATED, mode='r') as gen_f:
        for line in gen_f:
            generated.append(line.strip('\n'))

    summaries = list()
    with open(SUMMARIES, mode='r') as summ_f:
        for line in summ_f:
            summaries.append(line.strip('\n'))

    assert(len(generated) == len(summaries))
    log.info("Finished step: read data files.")

    log.info("Starting step: Evaluating " + str(len(generated)) + " summary pairs.")
    vert = ns.Vert()
    # generated = generated[len(generated)/4:]
    # summaries = summaries[len(summaries)/4:]
    scores = vert.score(
        generated,
        summaries,
        rouge_type=config['vert']['rouge_metric'],
        verbose=False
    )
    vert.display_scores(scores)
    if save_scores:
        vert.output_report(scores, config['vert']['reports_folder'])
    log.info("Finished step: Evaluate summary pairs")

if __name__ == '__main__':
    log.getLogger()
    if len(sys.argv) > 3:
        raise ValueError("Argument count too high.")
    if len(sys.argv) == 2:
        if sys.argv[1] == 'save':
            save_scores = True
        elif sys.argv[1] == 'toss':
            save_scores = False
        else:
            raise ValueError("Argument must be either 'save' or 'toss'")
        main(save_scores)
    else:
        main(config['vert']['output_report'])
