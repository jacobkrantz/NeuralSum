
from config import config
from ducArticle import DucArticle

from bs4 import BeautifulSoup
from collections import Counter
import nltk.data
from random import shuffle
import os
from numpy import asarray
"""
Provides utilities to load and preprocess all of the data used in the project.
Filename: 'preprocessing.py'
Methods:
    - parse_duc_2003()
    - parse_duc_2004()
    - display_articles(articles, number_to_display, random=False)
    - load_word_embeddings()
    - get_vocabulary(articles)
    - get_max_sentence_len(articles)
    - get_max_summary_len(articles)
    - get_sen_sum_pairs(articles)
    - fit_text(sentences, summaries, input_seq_max_length=None, target_seq_max_length=None)
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

def load_word_embeddings():
    """
    Reads all of the word embeddings (GloVe) into a dictionary.
    """
    embeddings = dict()
    with open(config['glove_embeddings_file'], 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32',)
            embeddings[word] = coefs

    return embeddings

def get_vocabulary(articles):
    """
    Decision here: what words to include in the vocabulary. We don't want to
        use the entire GloVe vector set, because the model would be too large.
    possibilities:
        - use sentences and gold summaries
        - use entire articles
        - use entire gigaword
        - use just sentences
    currently doing: use sentences and gold summaries
    Returns:
        set<string> vocab
    """
    vocab = set()
    for article in articles:
        [vocab.add(w) for w in article.sentence.split()]
        for summary in article.gold_summaries:
            [vocab.add(w) for w in summary.split()]

    return vocab

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


def fit_text(sentences, summaries, input_seq_max_length=None, target_seq_max_length=None):
    """
    This function was adapted from chen 0040 in the GitHub repository:
        https://github.com/chen0040/keras-text-summarization
    Args:
        sentences (list<string>)
        summaries (list<string>)
        input_seq_max_length (int): defaults to config value.
        target_seq_max_length (int): defaults to config value.
    Returns:
        dict of model configuration.
    """
    if input_seq_max_length is None:
        input_seq_max_length = config["max_input_seq_length"]
    if target_seq_max_length is None:
        target_seq_max_length = config["max_target_vocal_size"]
    input_counter = Counter()
    target_counter = Counter()
    max_input_seq_length = 0
    max_target_seq_length = 0

    for line in sentences:
        text = [word.lower() for word in line.split(' ')]
        seq_length = len(text)
        if seq_length > input_seq_max_length:
            text = text[0:input_seq_max_length]
            seq_length = len(text)
        for word in text:
            input_counter[word] += 1
        max_input_seq_length = max(max_input_seq_length, seq_length)

    for line in summaries:
        line2 = 'START ' + line.lower() + ' END'
        text = [word for word in line2.split(' ')]
        seq_length = len(text)
        if seq_length > target_seq_max_length:
            text = text[0:target_seq_max_length]
            seq_length = len(text)
        for word in text:
            target_counter[word] += 1
            max_target_seq_length = max(max_target_seq_length, seq_length)

    input_word2idx = dict()
    for idx, word in enumerate(input_counter.most_common(config['max_input_vocab_size'])):
        input_word2idx[word[0]] = idx + 2
    input_word2idx['PAD'] = 0
    input_word2idx['UNK'] = 1
    input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])

    target_word2idx = dict()
    for idx, word in enumerate(target_counter.most_common(config['max_target_vocal_size'])):
        target_word2idx[word[0]] = idx + 1
    target_word2idx['UNK'] = 0

    target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

    return {
        'input_word2idx': input_word2idx,
        'input_idx2word': input_idx2word,
        'target_word2idx': target_word2idx,
        'target_idx2word': target_idx2word,
        'num_input_tokens': len(input_word2idx),
        'num_target_tokens': len(target_word2idx),
        'max_input_seq_length': max_input_seq_length,
        'max_target_seq_length': max_target_seq_length
    }





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
                gold_sums.append(f.read().lstrip())

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
    replace all numbers with ... something...
    """
    sen = sentence.lower().replace('\n', ' ')
    sen = sen.replace(';', '').replace("'", '').replace('"', '')
    sen = sen.replace('?', ' ?').replace('. ', ' ').replace(', ', ' ')
    # sen = sen[:-1] if sen[-1] == '.' else sen
    return sen
