from evaluation import *
import NeuralSum as ns

def evaluate_examples():
    # one word changed that has a similar meaning
    generated = 'endeavour astronauts join two sections of international space station'
    target =    'endeavour astronauts join two segments of international space station'
    eval = ns.Evaluation()
    scores = eval.test_single(generated, target)
    print generated
    print target
    eval.display_scores(scores)

    # one word changed that alters the meaning of the sentence
    generated = 'endeavour astronauts remove two segments of international space station'
    target =    'endeavour astronauts join two segments of international space station'
    scores = eval.test_single(generated, target)
    print('')
    print generated
    print target
    eval.display_scores(scores)
    print('')

    # identical sentences: a baseline for comparison
    generated = 'endeavour astronauts join two segments of international space station'
    target =    'endeavour astronauts join two segments of international space station'
    scores = eval.test_single(generated, target)
    print('')
    print generated
    print target
    eval.display_scores(scores)
    print('')

    # 3.84 GB mem usage before start
    # 11.8 GB mem peak
