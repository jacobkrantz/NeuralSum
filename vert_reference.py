import NeuralSum as ns
from copy import deepcopy

"""
Compute an inter annotator VERT score.
"""

def main():
    articles = ns.parse_duc_2004()
    vert = ns.Vert()

    hyps = []
    refs = []

    for art in articles:
        for target in art.gold_summaries:
            gens = deepcopy(art.gold_summaries)
            gens.remove(target)
            for gen in gens:
                hyps.append(gen)
                refs.append(target)

    assert len(hyps) == len(refs)
    assert len(hyps) == 6000
    
    scores = vert.score(
        hyps,
        refs,
        rouge_type='recall',
        verbose=True
    )
    vert.display_scores(scores)
    vert.output_report(scores, './')

if __name__ == '__main__':
    main()
