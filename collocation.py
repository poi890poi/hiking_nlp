import os
from pathlib import Path
import logging
import pickle

import numpy as np
import nltk
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures
from gensim.models.phrases import Phrases, Phraser

from process import Vocabulary, ROOT_DIR


__VERSION__ = '0.1.3'


class Corpus():

    def __init__(self):
        self.__count = 0
    
    def __iter__(self):
        for p in Path(ROOT_DIR).rglob('*.html'):
            dir_, filename = os.path.split(p)
            prefix, extension = os.path.splitext(filename)
            pkl_path = os.path.join(dir_, '.'.join((prefix, 'pkl')))
            processed = None
            try:
                with open(pkl_path, 'rb') as fp:
                    processed = pickle.load(fp)
                    assert processed['__VERSION__'] == __VERSION__
            except (FileNotFoundError, AssertionError):
                pass
            if processed:
                for s in processed['tokenized']:
                    self.__count += 1
                    '''if self.__count % 1000 == 0:
                        print('Processing ngram... {} sentences processed'.format(self.__count))
                    s = s.astype(str)
                    yield s'''
                    for w in s:
                        self.__count += 1
                        if self.__count % 1000000 == 0:
                            print('Processing ngram... {} tokens processed'.format(self.__count))
                        yield w


if __name__ == '__main__':

    Vocabulary.load('./dataset/vocabulary.pkl')
    with open('./dataset/dictionary.tmp', 'w') as fp:
        for i in Vocabulary.most_common(4096):
            # 3188252 is the maximum frequency in jieba dictionary
            try:
                fp.write('{} {}\n'.format(Vocabulary.index2word()[i], int(Vocabulary.tfidf()[i] * 3188252)))
            except ValueError:
                pass

    corpus = Corpus()

    # There's no way to read phrases from Gensim Phrases model, making it less useful.
    '''bigram = Phrases(corpus, min_count=1, threshold=1)
    bigram.save('./dataset/bigram.pkl')
    exit()'''

    bigram_finder = BigramCollocationFinder.from_words(corpus)

    with open('./dataset/bigram.tmp', 'w') as fp:
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 4096)
        for bigram_tuple in bigrams:
            fp.write('{}\n'.format(Vocabulary.untokenize(bigram_tuple)))

    '''trigram_finder = TrigramCollocationFinder.from_words(corpus)

    with open('./dataset/trigram.tmp', 'w') as fp:
        trigrams = trigram_finder.nbest(TrigramAssocMeasures.chi_sq, 4096)
        for trigram_tuple in trigrams:
            fp.write('{}\n'.format(Vocabulary.untokenize(trigram_tuple)))'''
