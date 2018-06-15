from __future__ import print_function

import NeuralSum as ns

import os
"""
script for processing the DUC 2004 dataset for use in evaluation
    with tensor2tensor.
- extracts the raw articles into two txt files:
    - sentences.txt for the first sentence of each article.
    - summaries.txt for the target summary.
- line n of sentences.txt corresponds to line n of summaries.txt
- since there are about 4 summaries per sentence, there will be repeats in
    sentences.txt.
- utilizes NeuralSum/preprocessing.

make sure to rerun this script whenever preprocessing steps have been changed!
deletion of sentences.txt and summaries.txt is done automatically before.
"""

SENTENCES = './data/duc2004/sentences.txt'
SUMMARIES = './data/duc2004/summaries.txt'

def main():
    # remove existing files:
    try:
        os.remove(SENTENCES)
        print("Removed " + SENTENCES)
    except Exception as e:
        pass
    try:
        os.remove(SUMMARIES)
        print("Removed " + SUMMARIES)
    except Exception as e:
        pass

    # preprocess articles:
    duc2004articles = ns.parse_duc_2004()
    sentences_gen = sentence_generator(duc2004articles)
    summaries_gen = summary_generator(duc2004articles)

    # write sentences and summaries to repspective files:
    count_check = 0
    with open(SENTENCES, mode='a') as sen_f:
        for _, sen in enumerate(sentences_gen):
            sen_f.write(sen + '\n')
            count_check += 1

    with open(SUMMARIES, mode='a') as summ_f:
        for _, summ in enumerate(summaries_gen):
            summ_f.write(summ + '\n')
            count_check -= 1

    print(count_check)
    assert (count_check == 0), 'Sentence count did not equal summary count.'

def sentence_generator(articles):
    for art in articles:
        for _ in range(len(art.gold_summaries)):
            yield art.sentence

def summary_generator(articles):
    for art in articles:
        for summary in art.gold_summaries:
            yield summary

if __name__ == '__main__':
    main()
