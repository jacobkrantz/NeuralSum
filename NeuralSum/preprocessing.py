
from config import config
from ducArticle import DucArticle

from bs4 import BeautifulSoup
import nltk.data
from random import shuffle
import os
"""
Provides utilities to load and preprocess all of the data used in the project.
Filename: 'preprocessing.py'
Methods:
    - parse_duc_2003()
    - parse_duc_2004()
    - display_articles(articles, number_to_display, random=False)
"""

def parse_duc_2004():
    """
    Reads all of DUC-2004 into a single data structure.
    Eventually will probably need to tokenize everything.
    Returns:
        list<DucArticle>
    """
    return _add_duc_summaries_2004(_get_duc_sentences_2004())

def parse_duc_2003():
    """
    Reads all of DUC-2003 into a single data structure.
    Eventually will probably need to tokenize everything.
    Returns:
        list<DucArticle>
    """
    return _add_duc_summaries_2003(_get_duc_sentences_2003())

def display_articles(articles, number_to_display, random=False):
    if random:
        shuffle(articles)
    print "Contains", len(articles), "articles."
    for i in range(number_to_display):
        print articles[i]
        print ""



def _get_duc_sentences_2004():
    """
    Create a DucArticle for each article in the docs folder of Duc2004.
    Complete fields 'ID', 'folder', and 'sentence'.
    Returns:
        DucArticle
    """
    filenames = list()
    for root, _, files in os.walk(config["duc4_sentences_folder"], topdown=False):
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
            article.folder = filename.lstrip(config["duc4_sentences_folder"])[:5]
            article.sentence = sentence
            articles.append(article)

    return articles

def _add_duc_summaries_2004(articles):
    """
    Adds all gold standard summaries to each article by traversing the eval
        directory.
    Args:
        articles: list<DucArticle>
    Returns:
        list<DucArticle> with completed field 'gold_summaries'.
    """
    filenames = set()
    for root, _, files in os.walk(config["duc4_eval_folder"], topdown=False):
        for name in files:
            filenames.add(os.path.join(root, name))

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

    return articles

def _get_duc_sentences_2003():
    filenames = list()
    for root, _, files in os.walk(config["duc3_sentences_folder"], topdown=False):
        for name in files:
            filenames.append(os.path.join(root, name))

    articles = list()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for filename in filenames:
        with open(filename, 'r') as f:
            parsed_html = BeautifulSoup(f.read(), "lxml")
            if len(parsed_html.find_all('p')) > 0: # contains <P> ... </P>
                extracted = parsed_html.text.split('\n\n\n')[1].rstrip().lstrip().replace('\n', ' ')
                if len(extracted.split()) < 7: # hack: get rid of: "By firstname lastname A P writer"
                    extracted = parsed_html.text.split('\n\n\n')[2].rstrip().lstrip().replace('\n', ' ')
                sentences = tokenizer.tokenize(extracted)
            else: # formatted like duc2004
                extracted = parsed_html.find_all('text')[0].string
                sentences = tokenizer.tokenize(extracted)

            article = DucArticle()
            article.id = parsed_html.docno.string.rstrip().lstrip().replace('\n', ' ').encode('ascii','ignore')
            article.folder = filename.split('/')[4][:-1].upper()
            article.sentence = _tokenize_sentence(sentences)
            articles.append(article)

    return articles

def _add_duc_summaries_2003(articles):
    filenames = set()
    for root, _, files in os.walk(config["duc3_eval_folder"], topdown=False):
        for name in files:
            filenames.add(os.path.join(root, name))

    for article in articles:
        id = article.id
        folder = article.folder

        gold_summaries = []
        for filename in filenames:
            if (id not in filename) or (folder not in filename):
                continue
            with open(filename, 'r') as f:
                gold_summaries.append(f.read().lstrip().replace('\n',''))

        article.gold_summaries = gold_summaries

    return articles

def _tokenize_sentence(sentence):
    if sentence[0].split()[-1] not in config["duc_ending_exceptions"]:
        tokenized = sentence[0]
    else:
        tokenized = (sentence[0] + ' ' + sentence[1])

    # remove Newstation opening:
    if len(tokenized.split('--')) > 2:
        tokenized = "--".join(tokenized.split('--')[1:])
    else:
        tokenized = tokenized.split('--')[-1]
    if len(tokenized.split('_')) > 2:
        tokenized = "_".join(tokenized.split('_')[1:])
    else:
        tokenized = tokenized.split('_')[-1]
    return tokenized.encode('ascii','ignore').rstrip().lstrip().replace('\n',' ').replace('\t', ' ')
