
from config import config
import NeuralSum as ns
from InferSent import InferSent

from keras import backend as K
import logging as log
import numpy as np
import sys
from sklearn.model_selection import train_test_split

__author__ = 'jacobkrantz'
__copyright__ = 'jacobkrantz'
__license__ = 'none'
__version__ = '0.0.1'

def dev_test():
    log.info("Running developer script")
    hyps = [
        'hundreds of a voice in state state newspapers newspapers newspapers newspapers and jailed jailed opposition leader leader have turned to the internet to the internet',
        'indonesian president clinton of ``finds finds finds a summit of asia-pacific ``difficult because of ``asked his concerns about the arrest of malaysias former deputy deputy president',
        'among among asias leaders prime minister mahathir mohamad was mohamad as a man a bold: a physical and social social social social social that this party into the world affairs',
        'on on on the face of dissident anwar ibrahim on newspaper front front river for two days and from the the internet are are are unconstitutional and hundreds of malaysias'
    ]
    refs = [
        'anwar supporters speak out on internet unblocked by government',
        'regional leaders consider boycotting malaysian meeting due to anwar arrest',
        'mahathirs 17 years saw great advances now economic crisis instability',
        'malaysian prime minister expresses surprise at behavior of his police'
    ]
    vert = ns.Vert()
    # scores = vert.score(hyps, refs, verbose=True)
    scores = {
        'avg_hyp_word_cnt': 27.750,
        'approx_vert_score': 0.443,
        'avg_ref_word_cnt': 9.750,
        'wm_dist': 3.225,
        'cos_sim': 0.7027,
        'rouge-2': 0.000,
        'num_tested': 4,
        'rouge-1': 7.778,
        'test_type': 'recall',
        'vert_score': 0.443,
        'rouge-l': 7.778
    }
    vert.display_scores(scores)
    vert.output_report(scores, config["vert"]["reports_folder"])
    log.info("Done running developer script")

def train():
    log.info("Starting step: load training data.")
    duc_2003_articles = ns.parse_duc_2003()
    duc_2004_articles = ns.parse_duc_2004()
    num_articles = len(duc_2003_articles) + len(duc_2004_articles)
    log.info("Finished step: load training data. Articles: %s " % num_articles)

    log.info("Starting step: load word embeddings.")
    embeddings = ns.load_word_embeddings()
    log.info("Finished step: load word embeddings. Size: %s" % len(embeddings))

    log.info("Starting step: extract vocabulary.")
    vocab = ns.get_vocabulary(duc_2003_articles + duc_2004_articles)
    max_sen_len = ns.get_max_sentence_len(duc_2003_articles)
    max_sum_len = ns.get_max_summary_len(duc_2003_articles + duc_2004_articles)
    sentences, summaries = ns.get_sen_sum_pairs(duc_2003_articles)
    log.info("Finished step: extract vocabulary. Size: %s." %len(vocab))

    log.info("Starting step: construct model.")
    model = ns.SummaryModel(m_params=ns.fit_text(sentences, summaries))
    model.compile(
        embeddings,
        vocab,
        max_sen_len,
        max_sum_len
    )
    log.info("Finished step: construct model.")

    log.info("Starting step: fit model.")
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(sentences, summaries, test_size=0.2, random_state=39)
    history = model.train(Xtrain, Xtest, Ytrain, Ytest)
    # do some visualization of history
    K.clear_session()
    log.info("Finished step: fit model.")

def test(verbosity):
    log.info("Starting step: load testing data.")
    duc_2003_articles = ns.parse_duc_2003()
    duc_2004_articles = ns.parse_duc_2004()
    num_articles = len(duc_2004_articles)
    log.info("Finished step: load testing data. Articles: %s " % num_articles)

    log.info("Starting step: load word embeddings.")
    embeddings = ns.load_word_embeddings()
    log.info("Finished step: load word embeddings. Size: %s" % len(embeddings))

    log.info("Starting step: extract vocabulary.")
    vocab = ns.get_vocabulary(duc_2003_articles + duc_2004_articles)
    max_sen_len = ns.get_max_sentence_len(duc_2003_articles)
    max_sum_len = ns.get_max_summary_len(duc_2003_articles + duc_2004_articles)
    sentences, summaries = ns.get_sen_sum_pairs(duc_2003_articles)
    log.info("Finished step: extract vocabulary. Size: %s." %len(vocab))

    log.info("Starting step: load model.")
    m_params = np.load(
        config['model_output_path'] + config['model_config_file']
    ).item()
    model = ns.SummaryModel(m_params)
    model.compile(
        embeddings,
        vocab,
        max_sen_len,
        max_sum_len
    )
    model.load_weights(
        config['model_output_path'] + config['model_weight_file']
    )
    log.info("Finished step: load model.")

    log.info("Starting step: generate summaries.")
    num_to_test = -1
    duc_2004_articles = model.test(duc_2004_articles, num_to_test=num_to_test)
    if num_to_test > 0:
        ns.display_articles(duc_2004_articles)
    log.info("Finished step: generate summaries.")

    log.info("Starting step: calulate ROUGE and InferSent scores.")
    vert = ns.Vert()
    scores = vert.score_from_articles(
        duc_2004_articles,
        rouge_type=config['vert']['rouge_metric'],
        verbose=False
    )
    if verbosity == 1:
        vert.display_scores(scores)
    if config['vert']['output_report']:
        vert.output_report(scores, config['vert']["reports_folder"])
    log.info("Finished step: calulate ROUGE and InferSent scores.")

def main():
    if len(sys.argv) == 1:
        dev_test()
        return

    arg_index = 1
    # flags for log verbosity
    verbosity = 1
    if sys.argv[arg_index] == '-v':
        log.getLogger().setLevel(log.DEBUG)
        arg_index += 1
    elif sys.argv[arg_index] == '-s':
        log.getLogger().setLevel(log.CRITICAL)
        verbosity = 0
        arg_index += 1

    # arguments for actions to perform
    if sys.argv[arg_index] == 'train':
        train()
    elif sys.argv[arg_index] == 'test':
        test(verbosity)
    else:
        raise SyntaxError("Bad argument(s). flags: '-v' or '-s' args: 'train' or 'test'")

if __name__ == '__main__':
    if len(config) > 0:
        log.info("Configuration loaded.")
        main()
    else:
        log.critical("Configuration failed to load.")
