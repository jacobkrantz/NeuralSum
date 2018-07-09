
import logging as log
from NeuralSum import Vert, parse_duc_2004
from rouge import Rouge
import sys

"""
script for evaluating the inter-annotator aggreement in DUC2004.
Compares each target summary with each other for a given article.
makes these comparisons:
    (3 tests) * (4 targets) * (0.5) * (500 articles) = 3000 comparisons
    multiply by 0.5 because ordering does not matter for comparison.

ROUGE:
if no args specified, computes 'fast' version of ROUGE. ROUGE actually has an
order dependency. If you want to run full rouge by itself, call:
>>> python inter_annotator_agreement.py rouge
"""

def make_sets():
    log.info("Starting: loading DUC datasets")
    hyps = []
    refs = []
    duc_articles = parse_duc_2004()
    for art in duc_articles:
        s = art.gold_summaries
        # manually permute. Not the greateset.
        hyps += [s[0],s[0],s[0]]
        refs += [s[1],s[2],s[3]]

        hyps += [s[1],s[1],s[2]]
        refs += [s[2],s[3],s[3]]

    log.info("Finished: Loading DUC datasets")
    return hyps, refs

def score_sets(hyps, refs):
    log.info("Starting step: Evaluating " + str(len(refs)) + " summary pairs.")
    assert(len(hyps) == len(refs))

    vert = Vert()
    scores = vert.score(
        hyps,
        refs,
        rouge_type='recall',
        verbose=True
    )
    vert.display_scores(scores)
    vert.output_report(scores, "./data/out/vert_reports/")

    del vert
    log.info("Finished step: Evaluate summary pairs.")
    return scores

def score_rouge_full(hyps,refs):
    """Gives a true ROUGE score unlike the other, which is an approx."""
    assert(len(hyps) == len(refs))
    rouge = Rouge()
    r_scores = rouge.get_scores(hyps, refs, avg=True)
    rouge_1 = r_scores['rouge-1']['r'] * 100
    rouge_2 = r_scores['rouge-2']['r'] * 100
    rouge_l = r_scores['rouge-l']['r'] * 100
    print 'rouge-1:'
    print rouge_1
    print 'rouge-2:'
    print rouge_2
    print 'rouge-l:'
    print rouge_l

def rouge():
    log.info("Starting: loading DUC datasets")
    hyps = []
    refs = []
    duc_articles = parse_duc_2004()
    for art in duc_articles:
        s = art.gold_summaries
        # manually permute. Not the greateset.
        hyps += [s[0],s[0],s[0]]
        refs += [s[1],s[2],s[3]]

        hyps += [s[1],s[1],s[1]]
        refs += [s[0],s[2],s[3]]

        hyps += [s[2],s[2],s[2]]
        refs += [s[1],s[0],s[3]]

        hyps += [s[3],s[3],s[3]]
        refs += [s[1],s[2],s[0]]
    log.info("Finished: Loading DUC datasets")
    score_rouge_full(hyps, refs)

def main():
    hyps, refs = make_sets()
    scores = score_sets(hyps, refs)

if __name__ == '__main__':
    log.getLogger()
    if len(sys.argv) == 2:
        if sys.argv[1] != 'rouge':
            raise ValueError("only argument valid: rouge")
        else:
            rouge()
    elif len(sys.argv) > 2:
        raise ValueError("only one argument valid: rouge")
    else:
        main()
