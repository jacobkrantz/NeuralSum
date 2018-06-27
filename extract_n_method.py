
import NeuralSum as ns
import numpy as np
import matplotlib.pyplot as plt
from rouge import Rouge

"""
Takes the most simple summarization technique: copy the first n words from
    the input sequence that keeps the length of the summary less than 75 chars.

Results:
Gigaword:
    {'rouge-1': '35.704', 'rouge-2': '12.208', 'rouge-l': '32.979'}
    word counts under 75 chars
        max:        26
        min:        6
        average:    13.0385
DUC2004:
    {'rouge-1': '22.043', 'rouge-2': '6.199', 'rouge-l': '19.641'}
    word counts under 75 chars
        max:        18
        min:        5
        average:    12.884
DUC2003:
    {'rouge-1': '21.042', 'rouge-2': '5.757', 'rouge-l': '18.696'}
    word counts under 75 chars
        max:        19
        min:        3
        average:    12.880
Takeaways:
    - Copying words directly performs 13 points better ROUGE-1 on Gigaword
        than on DUC2004. This could be related to why training on Gigaword
        learned to do this. 32.979 ROUGE-1 is high!
    - I need to show why this technique is bad though examples.
    - I need my method to beat 22.043 ROUGE-1 in order to be moderately viable.
"""

def main():
    articles = ns.parse_duc_2003()
    # articles = ns.parse_duc_2004()
    # articles = ns.parse_gigaword()
    print 'loaded articles.'

    gen_target_pairs = []
    for i, art in enumerate(articles):
        gen_s = make_summary(art.sentence)
        map(lambda s: gen_target_pairs.append((s,gen_s)), art.gold_summaries)
        if len(articles) % (i+1) == 0:
            print len(articles), i+1

    w_cnts = list(map(lambda p: len(p[1].split()), gen_target_pairs))
    print "Word Counts (under 75 characters):"
    print "Max:", max(w_cnts), "Min:", min(w_cnts), "Avg:", np.mean(w_cnts)

    # # plots the word counts for all summaries:
    # plt.hist(word_counts, bins=30)
    # plt.xlabel('Word Count: Under 75 Chars');
    # plt.show()

    rouge = Rouge()
    r_scores = rouge.get_scores(
        hyps=list(map(lambda p: p[1], gen_target_pairs)),
        refs=list(map(lambda p: p[0], gen_target_pairs)),
        avg=True
    )
    print {
        'rouge-1':"{0:.3f}".format(r_scores['rouge-1']['r'] * 100),
        'rouge-2':"{0:.3f}".format(r_scores['rouge-2']['r'] * 100),
        'rouge-l':"{0:.3f}".format(r_scores['rouge-l']['r'] * 100)
    }

def make_summary(sent):
    if len(sent) < 75:
        return sent

    sent = sent.split()
    summary = ""
    n = 3
    while len(summary) < 75:
        summary = " ".join(sent[:n])
        n += 1

    return summary

if __name__ == '__main__':
    main()
