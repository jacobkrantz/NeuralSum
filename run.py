
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
    log.info("Running Developer Script")
    duc_2004_articles = ns.parse_duc_2004()
    duc_2003_articles = ns.parse_duc_2003()
    vocab =  ns.get_vocabulary(duc_2003_articles + duc_2004_articles)

    # display_articles(duc_2003_articles, 3, random=False)
    duc_2004_articles[0].generated_summary = "supporters of Malaysia's opposition leader speak out"
    duc_2004_articles[1].generated_summary = "Habibie struggles to attend Asia-Pacific summit"
    scores = ns.test_all_articles([duc_2004_articles[0], duc_2004_articles[1]])
    ns.display_scores(scores)

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

    log.info("Starting step: calulate ROUGE scores.")
    scores = ns.test_all_articles(duc_2004_articles)
    if verbosity == 1:
        ns.display_scores(scores)
    log.info("Finished step: calulate ROUGE scores.")

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
    log.getLogger('')
    if len(config) > 0:
        log.info("Configuration loaded.")
        main()
    else:
        log.critical("Configuration failed to load.")
