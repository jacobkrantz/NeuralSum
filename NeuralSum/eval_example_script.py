from evaluation import *
import NeuralSum as ns

def evaluate_examples():
    generated = 'endeavour astronauts join two sections of international space station'
    target = 'endeavour astronauts join two segments of international space station'

    eval = ns.Evaluation()
    print generated
    print target
    scores = eval.test_single(generated, target)
    eval.display_scores(scores)

    generated = 'endeavour astronauts remove two segments of international space station'
    target = 'endeavour astronauts join two segments of international space station'

    eval = ns.Evaluation()
    print generated
    print target
    scores = eval.test_single(generated, target)
    eval.display_scores(scores)
