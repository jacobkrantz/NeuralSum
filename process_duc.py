import fire
import logging as log
import os

import NeuralSum as ns

"""
script for processing DUC datasets (2003 or 2004) for use in evaluation
    with tensor2tensor.
- extracts the raw articles into two txt files:
    - sentences.txt for the first sentence of each article.
    - summaries.txt for the target summary.
- line n of sentences.txt corresponds to line n of summaries.txt
- since there are 4 summaries per sentence, there will be repeats in
    sentences.txt.
- utilizes NeuralSum/preprocessing.
- defaults to processing DUC2004 if no argument is provided.
- run options:
    >>> python process_duc.py [--which_duc=(2003|2004)]

make sure to rerun this script whenever preprocessing steps have been changed!
deletion of sentences.txt and summaries.txt is done automatically.
"""

def main(which_duc=2004):
    if type(which_duc) != int or which_duc not in [2003, 2004]:
        msg = 'which_duc must be either 2003 or 2004.'
        log.exception(msg)
        raise ValueError(msg)

    sentences = './data/duc' + str(which_duc) + '/sentences.txt'
    summaries = './data/duc' + str(which_duc) + '/summaries.txt'

    # remove existing files:
    try:
        os.remove(sentences)
        log.info("Removed " + sentences)
    except Exception as e:
        pass
    try:
        os.remove(summaries)
        log.info("Removed " + summaries)
    except Exception as e:
        pass

    # preprocess articles:
    log.info('Loading DUC' + str(which_duc) + ' articles')
    duc_articles = ns.parse_duc_2004() if which_duc == 2004 else ns.parse_duc_2003()

    # write sentences and summaries to repspective files:
    sent_count = 0
    with open(sentences, mode='a') as sen_f:
        for _, sen in enumerate(sentence_generator(duc_articles)):
            sen_f.write(sen + '\n')
            sent_count += 1
        log.info("Wrote " + str(sent_count) + " sentences to: " + sentences)

    summ_count = 0
    with open(summaries, mode='a') as summ_f:
        for _, summ in enumerate(summary_generator(duc_articles)):
            summ_f.write(summ + '\n')
            summ_count += 1
        log.info("Wrote " + str(summ_count) + " summaries to: " + summaries)

    assert (summ_count == sent_count), 'Sentence count did not equal summary count.'

def sentence_generator(articles):
    for art in articles:
        for _ in range(len(art.gold_summaries)):
            yield art.sentence

def summary_generator(articles):
    for art in articles:
        for summary in art.gold_summaries:
            yield summary

if __name__ == '__main__':
    log.getLogger()
    fire.Fire(main)
