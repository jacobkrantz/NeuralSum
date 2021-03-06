from __future__ import print_function

from config import config
from duc_article import DucArticle

import numpy as np
from bs4 import BeautifulSoup
from collections import Counter
import nltk.data
from nltk import download
from nltk.downloader import Downloader
from random import shuffle
import os
import re
from numpy import asarray
"""
Provides utilities to load and preprocess all of the data used in the project.
Filename: 'preprocessing.py'
Methods:
    - parse_duc_2004(limit=None, randomize=False)
    - parse_duc_2003(limit=None, randomize=False)
    - parse_gigaword(limit=None, randomize=False)
    - get_max_sentence_len(articles)
    - get_max_summary_len(articles)
    - get_sen_sum_pairs(articles)
    - display_articles(articles, number_to_display, random=False)
"""

def parse_duc_2004(limit=None, randomize=False):
    """
    Reads all of DUC-2004 into a single data structure.
    Args:
        limit (int): caps the number of articles to return. Default to all.
        randomize (bool): should the articles be chosen and returned randomly.
    Returns:
        list<DucArticle>
    """
    articles = _add_duc_summaries_2004(_get_duc_sentences_2004())
    if randomize:
        np.random.shuffle(articles)

    return articles if limit is None else articles[:limit]

def parse_duc_2003(limit=None, randomize=False):
    """
    Reads all of DUC-2003 into a single data structure.
    Args:
        limit (int): caps the number of articles to return. Default to all.
        randomize (bool): should the articles be chosen and returned randomly.
    Returns:
        list<DucArticle>
    """
    articles = _add_duc_summaries_2003(_get_duc_sentences_2003())
    if randomize:
        np.random.shuffle(articles)

    return articles if limit is None else articles[:limit]

def parse_gigaword(limit=None, randomize=False):
    """
    Reads all 3.8 million gigaword sentence-summary pairs into the
    DucArticle format for use in training models.
    Args:
        limit (int): caps the number of articles to return. Default to all.
        randomize (bool): should the articles be chosen and returned randomly.
    """
    articles = _add_gigaword_summaries(_get_gigaword_sentences())
    if randomize:
        np.random.shuffle(articles)

    return articles if limit is None else articles[:limit]

def get_max_sentence_len(articles):
    """
    Get the maximum length of full sentence within the articles list.
    Args:
        articles (list<DucArticle>)
    Returns:
        int length
    """
    return max(map(lambda a: len(a.sentence.split()), articles))

def get_max_summary_len(articles):
    """
    Get the maximum length of a summary within the articles list.
    Args:
        articles (list<DucArticle>)
    Returns:
        int length
    """
    return max([
            max(map(lambda s: len(s.split()), art.gold_summaries))
            for art in articles
    ])

def get_sen_sum_pairs(articles):
    """
    Returns two lists where the index relates the sentence to the summary.
    Args:
        articles (list<DucArticle>)
    Returns:
        list<string> sentences, list<string> summaries
    """
    sentences = list()
    summaries = list()

    for article in articles:
        for sum in article.gold_summaries:
            sentences.append(article.sentence)
            summaries.append(sum)

    assert(len(sentences) == len(summaries))
    return sentences, summaries

def display_articles(articles, number_to_display=None, randomize=False):
    """
    If number_to_display is None, display all.
    If random is True, shuffle articles before selecting and displaying.
    """
    if randomize:
        np.random.shuffle(articles)
    if number_to_display is not None:
        articles = articles[:number_to_display]
    map(lambda art: print(art, '\n'), articles)

def _get_duc_sentences_2004():
    """
    Create a DucArticle for each article in the docs folder of Duc2004.
    Complete fields 'ID', 'folder', and 'sentence'.
    Returns:
        list<DucArticle>
    """
    if not Downloader().is_installed('punkt'):
        download('punkt')

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
            article.sentence = _tokenize_sentence_generic(sentence)
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

        gold_sums = []
        for filename in filenames:
            if (id not in filename) or (folder not in filename):
                continue
            with open(filename, 'r') as f:
                gold_sums.append(f.read())

        article.gold_summaries = [_tokenize_sentence_generic(s) for s in gold_sums]

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
            article.sentence = _tokenize_sentence_generic(
                _tokenize_sentence_2003(sentences)
            )
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

        gold_sums = []
        for filename in filenames:
            if (id not in filename) or (folder not in filename):
                continue
            with open(filename, 'r') as f:
                gold_sums.append(f.read().lstrip().replace('\n',''))

        article.gold_summaries = [_tokenize_sentence_generic(s) for s in gold_sums]

    return articles

def _get_gigaword_sentences():
    """
    get all of the sentences from the gigaword train data.
    """
    articles = list()
    with open(config['gigaword_sen_folder'], 'r') as f:
        for line in f:
            art = DucArticle()
            art.sentence = line.strip('\n')
            articles.append(art)
    return articles

def _add_gigaword_summaries(articles):
    """
    get all of the summaries from the gigaword train data. Add them
        to the article gold_summaries list.
    """
    summaries = list()
    with open(config['gigaword_sum_folder'], 'r') as f:
        for line in f:
            summaries.append(line)

    assert(len(summaries) == len(articles))
    for i, summary in enumerate(summaries):
        articles[i].gold_summaries = [summary.strip('\n')]
    return articles

def _tokenize_sentence_2003(sentence):
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

def _tokenize_sentence_generic(sentence):
    """
    space out 's and contractions isn't -> is n't or -> isnt
    space out all punctuation
    remove quotations
    replace all numbers with '#'
    """
    # strip left and right whitespace
    sen = sentence.lstrip().rstrip()

    sen = sen.lower().replace('\n', ' ').replace("''", " ").replace("``", " ")
    sen = sen.replace("'s", " 's").replace("'re", " 're")
    sen = sen.replace("``", '').replace('(', ' ').replace(')', ' ').replace('*', ' ')
    sen = sen.replace(';', '').replace('"', '').replace('_', ' ').replace(':', ' :')
    sen = sen.replace('?', ' ?').replace(', ', ' , ').replace('!', ' !')

    # replace numbers with '#'
    sen = re.sub('\d', '#', sen)

    # space the ending '.'
    sen = sen[:-1] + ' .' if sen[-1] == '.' else sen

    # remove duplicate whitespace
    sen = " ".join(sen.split())

    # space common contractions. ignore won't.
    sen = sen.replace("isn't", "is n't")
    sen = sen.replace("hasn't", "has n't")
    sen = sen.replace("doesn't", "does n't")
    sen = sen.replace("don't", "do n't")
    sen = sen.replace("wouldn't", "would n't")
    return sen
