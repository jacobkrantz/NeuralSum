import fire
import logging as log
import matplotlib.pyplot as plt
import numpy as np

import NeuralSum as ns

"""
Testing the length of Gigaword article summaries
"""

def main():

    # preprocess articles:
    log.info('Loading Gigaword articles')
    giga_articles = ns.parse_gigaword()
    log.info('Done: loading Gigaword articles')

    sum_lengths = []
    for art in giga_articles:
        sum_lengths += list(map(lambda s: len(s.split()), art.gold_summaries))

    count_11 = 0
    for art in giga_articles:
        l = len(art.gold_summaries[0].split())
        if l < 11:
            count_11 += 1

    print 'count_11:', count_11

    new_arts = []
    for art in giga_articles:
        if len(art.gold_summaries[0].split()) > 10:
            new_arts.append(art)

    avg_l = np.mean(list(map(lambda s: len(s.gold_summaries[0].split()), new_arts)))
    print 'original len: ', len(giga_articles)
    print 'new len: ', len(new_arts)
    print 'avg len with summaries less than len 10 removed:', avg_l

    # plots the word counts for all summaries:
    plt.hist(sum_lengths, bins=30)
    plt.xlabel('Word Count');
    plt.show()

if __name__ == '__main__':
    log.getLogger()
    fire.Fire(main)
