from __future__ import print_function

import NeuralSum as ns

import logging as log
import os
import sys

"""
script for processing DUC datasets (2003 or 2004) for use in evaluation
    with tensor2tensor.
- extracts the raw articles into two txt files:
    - sentences.txt for the first sentence of each article.
    - summaries.txt for the target summary.
- line n of sentences.txt corresponds to line n of summaries.txt
- since there are about 4 summaries per sentence, there will be repeats in
    sentences.txt.
- utilizes NeuralSum/preprocessing.
- run options:
    >>> python process_duc.py 2003
    >>> python process_duc.py 2004

make sure to rerun this script whenever preprocessing steps have been changed!
deletion of sentences.txt and summaries.txt is done automatically.
"""

def main(sentences, summaries):
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
    if '2004' in sentences:
        log.info("Loading DUC2004 articles")
        duc_articles = ns.parse_duc_2004()
    else:
        log.info("Loading DUC2003 articles")
        duc_articles = ns.parse_duc_2003()

    sentences_gen = sentence_generator(duc_articles)
    summaries_gen = summary_generator(duc_articles)

    # write sentences and summaries to repspective files:
    sent_count = 0
    with open(sentences, mode='a') as sen_f:
        for _, sen in enumerate(sentences_gen):
            sen_f.write(sen + '\n')
            sent_count += 1
        log.info("Wrote " + str(sent_count) + " sentences to: " + sentences)

    summ_count = 0
    with open(summaries, mode='a') as summ_f:
        for _, summ in enumerate(summaries_gen):
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
    if len(sys.argv) > 3:
        raise ValueError("Argument count too high.")
    if len(sys.argv) == 2:
        if sys.argv[1] == '2004':
            sentences = './data/duc2004/sentences.txt'
            summaries = './data/duc2004/summaries.txt'
        elif sys.argv[1] == '2003':
            sentences = './data/duc2003/sentences.txt'
            summaries = './data/duc2003/summaries.txt'
        else:
            raise ValueError("Argument must be either '2004' or '2003'")
        main(sentences, summaries)
    else:
        main('./data/duc2004/sentences.txt','./data/duc2004/summaries.txt')
