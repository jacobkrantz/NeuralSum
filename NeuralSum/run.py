
from config import config
from evaluation import test_article, test_all_articles
from preprocessing import parse_duc_2004, parse_duc_2003, display_articles, load_word_embeddings, get_vocabulary

import logging as log
import sys

def dev_test():
    # preprocessing tasks
    duc_2004_articles = parse_duc_2004()
    duc_2003_articles = parse_duc_2003()
    # display_articles(duc_2003_articles, 3, random=False)

    duc_2004_articles[0].generated_summary = "supporters of Malaysia's opposition leader speak out"
    duc_2004_articles[1].generated_summary = "Habibie struggles to attend Asia-Pacific summit"

    print test_article(duc_2004_articles[0], type=config['rouge']['reporting_metric'])
    print test_article(duc_2004_articles[1], type=config['rouge']['reporting_metric'])
    print test_all_articles(duc_2004_articles[0:2], type=config['rouge']['reporting_metric'])

    # h = "Cambodian government rejects opposition 's call for talks abroad"
    # r = "Cambodian leader Hun Sen rejects opposition demands for talks in Beijing."
    # test_single(h,r)
    # print test_single(h,r)
    # print('')
    # test_all([h,h1], [r,r1])

def train():
    log.info("Starting step: load testing data")

    duc_2003_articles = parse_duc_2003()
    duc_2004_articles = parse_duc_2004()
    log.debug("Starting step: load word embeddings.")

    embeddings = load_word_embeddings()

    vocab, vocab_size = get_vocabulary(duc_2003_articles + duc_2004_articles)
    # log.debug("Finished step: loaded %s word embeddings." % len(embeddings))
    log.info("Finished step: load testing data")

    # construct model
    # train model with data
    # save model & weights to files

def test():
    # load model
    # load testing data
    log.info("Starting step: load testing data")
    duc_2004_articles = parse_duc_2004()
    log.info("Finished step: load testing data")
    # predict all summaries
    # perform ROUGE tests
    raise NotImplementedError('no model built yet')

def main():
    if len(sys.argv) == 1:
        dev_test()
        return

    arg_index = 1
    # flags for log verbosity
    if sys.argv[arg_index] == '-v':
        log.getLogger().setLevel(log.DEBUG)
        arg_index += 1
    elif sys.argv[arg_index] == '-s':
        log.getLogger().setLevel(log.CRITICAL)
        arg_index += 1

    # arguments for actions to perform
    if sys.argv[arg_index] == 'train':
        train()
    elif sys.argv[arg_index] == 'test':
        test()
    else:
        raise SyntaxError("Bad argument(s). flags: '-v' or '-s' args: 'train' or 'test'")

if __name__ == '__main__':
    log.getLogger('')
    if len(config) > 0:
        log.info("Configuration loaded.")
        main()
    else:
        log.critical("Configuration failed to load.")
