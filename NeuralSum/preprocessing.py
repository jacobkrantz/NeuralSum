
from config import config
from ducArticle import DucArticle

from bs4 import BeautifulSoup
import nltk.data
import os

def parse_duc():
    """
    Reads all of DUC-2004 into a single data structure.
    Eventually will probably need to tokenize everything.
    Returns:
        list<DucArticle>
    """
    return _add_duc_summaries(_get_duc_sentences())

def _get_duc_sentences():
    filenames = list()
    for root, _, files in os.walk(config["duc_sentences_folder"], topdown=False):
        for name in files:
            filenames.append(os.path.join(root, name))

    articles = list()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for filename in filenames:
        with open(filename, 'r') as f:
            parsed_html = BeautifulSoup(f.read(), "lxml")
            corpus = parsed_html.find_all('text')[0].string
            tokenized = tokenizer.tokenize(corpus)
            if tokenized[0].split()[-1] not in config["duc_ending_exceptions"]:
                sentence = tokenized[0].encode('ascii','ignore')
            else:
                sentence = (tokenized[0] + ' ' + tokenized[1]).encode('ascii','ignore')

            article = DucArticle()
            article.id = parsed_html.docno.string.rstrip().lstrip().replace('\n', ' ').encode('ascii','ignore')
            article.folder = filename.lstrip(config["duc_sentences_folder"])[:5]
            article.sentence = sentence
            articles.append(article)
    return articles

def _add_duc_summaries(articles):
    filenames = set()
    for root, _, files in os.walk(config["duc_eval_folder"], topdown=False):
        for name in files:
            filenames.add(os.path.join(root, name))

    new_articles = []
    for article in articles:
        id = article.id
        folder = article.folder

        gold_summaries = []
        for filename in filenames:
            if (id not in filename) or (folder not in filename):
                continue
            with open(filename, 'r') as f:
                gold_summaries.append(f.read().lstrip())

        article.gold_summaries = gold_summaries

    # for j in range(3):
    #     print articles[j].sentence
    #     for i in articles[j].gold_summaries:
    #         print '\t' + i
    return new_articles
