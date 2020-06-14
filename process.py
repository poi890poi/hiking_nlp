import os
from pathlib import Path
import logging
import pickle
from collections import Counter

import numpy as np
import jieba
import nltk
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures
import lxml
import lxml.html
from lxml.html.clean import Cleaner


__VERSION__ = '0.1.4'
__LOG_LEVEL = logging.WARN
ROOT_DIR = './dataset/keepon/articles'

jieba.set_dictionary('./dataset/dict.txt')
jieba.load_userdict('./dataset/userdict.txt')

class Vocabulary():

    __word2index = dict()
    __index2word = np.ndarray((0,), dtype=object)

    __document_count = 0
    __term_frequency = np.ndarray((0,), dtype=np.float)
    __doc_frequency = np.ndarray((0,), dtype=np.uintc)
    __tfidf = np.ndarray((0,))

    @classmethod
    def word2index(cls):
        return cls.__word2index
    
    @classmethod
    def index2word(cls):
        return cls.__index2word

    @classmethod
    def word_frequency(cls):
        return cls.__term_frequency

    @classmethod
    def tfidf(cls):
        return cls.__tfidf

    @classmethod
    def __process_tfidf(cls):
        print('Processing TFIDF')
        cls.__term_frequency /= np.sum(cls.__term_frequency)
        cls.__doc_frequency = np.log(cls.__document_count / cls.__doc_frequency)
        cls.__tfidf = cls.__term_frequency * cls.__doc_frequency
        print('TFIDF processed', [(cls.__index2word[i], cls.__tfidf[i]) for i in cls.most_common()])

    @classmethod
    def most_common(cls, n=32):
        return np.argpartition(cls.__tfidf, -n)[-n:]

    @classmethod
    def add_words(cls, sentence):
        for word in sentence:
            try:
                windex = cls.__word2index[word]
            except KeyError:
                windex = len(cls.__word2index)
                cls.__word2index[word] = windex
                cls.__index2word.resize((len(cls.__index2word) + 1,))
                cls.__index2word[-1] = word
            cls.__term_frequency.resize((len(cls.__word2index),))
            cls.__term_frequency[windex] += 1

    @classmethod
    def add_document(cls, document):
        tokens = tuple(w for s in document for w in s)
        if tokens:
            counter = Counter(tokens)
            _, max_count = counter.most_common(1)[0]
            unique_tokens = np.array(tuple(set(tokens)), dtype=np.uintc)
            cls.__doc_frequency.resize(len(cls.__term_frequency))
            for word_index in unique_tokens:
                # augmented term frequency is used to prevent bias toward longer documents
                aug_freq = 0.5 + 0.5 * counter[word_index] / max_count
                #print(cls.__index2word[word_index], aug_freq)
                cls.__doc_frequency[word_index] += 1
                cls.__term_frequency[word_index] += aug_freq
            cls.__document_count += 1

    @classmethod
    def tokenize_sentence(cls, sentence):
        sentence = np.array([cls.__word2index[word] for word in sentence], dtype=np.uintc)
        return sentence

    @classmethod
    def untokenize(cls, sentence):
        sentence = np.array([cls.__index2word[index] for index in sentence], dtype=object)
        return sentence

    @classmethod
    def load(cls, filename):
        cls.__filename = filename
        try:
            with open(cls.__filename, 'rb') as fp:
                vocabulary = pickle.load(fp)
            assert vocabulary['__VERSION__'] == __VERSION__
            cls.__word2index = vocabulary['word2index']
            cls.__index2word = np.copy(vocabulary['index2word'])
            cls.__tfidf = np.copy(vocabulary['index2word'])
            cls.__term_frequency = np.copy(vocabulary['word_frequency'])
            cls.__document_count = vocabulary['document_count']
            cls.__doc_frequency = np.copy(vocabulary['doc_frequency'])
            cls.__tfidf = np.copy(vocabulary['tfidf'])
            try:
                if __LOG_LEVEL == logging.DEBUG:
                    assert False
            except NameError:
                pass
            print('Vocabulary loaded')
            del vocabulary
        except (FileNotFoundError, KeyError, AssertionError):
            cls.add_words(['__NULL__',])
            print('Corpus or processings changed; vocabulary need re-process')

    @classmethod
    def reset_frequency(cls):
        cls.__document_count = 0
        cls.__term_frequency = np.ndarray((0,), dtype=np.float)
        cls.__doc_frequency = np.ndarray((0,), dtype=np.uintc)
        cls.__tfidf = np.ndarray((0,))

    @classmethod
    def save(cls):
        cls.__process_tfidf()
        vocabulary = {
            '__VERSION__': __VERSION__,
            'word2index': cls.__word2index,
            'index2word': cls.__index2word,
            'word_frequency': cls.__term_frequency,
            'document_count': cls.__document_count,
            'doc_frequency': cls.__doc_frequency,
            'tfidf': cls.__tfidf,
        }
        with open(cls.__filename, 'wb') as fp:
            pickle.dump(vocabulary, fp)
        print('Vocabulary saved at {}'.format(cls.__filename))


if __name__ == '__main__':

    Vocabulary.load('./dataset/vocabulary.pkl')
    Vocabulary.reset_frequency()

    count = 0
    cleaner = Cleaner()
    cleaner.javascript = True # This is True because we want to activate the javascript filter
    cleaner.style = True      # This is True because we want to activate the styles & stylesheet filter

    corpus = list()

    for p in Path(ROOT_DIR).rglob('*.html'):
        dir_, filename = os.path.split(p)
        prefix, extension = os.path.splitext(filename)
        pkl_path = os.path.join(dir_, '.'.join((prefix, 'pkl')))
        processed = None
        print(count, pkl_path)
        try:
            with open(pkl_path, 'rb') as fp:
                processed = pickle.load(fp)
                assert processed['__VERSION__'] == __VERSION__
            if __LOG_LEVEL == logging.DEBUG:
                assert False
        except (FileNotFoundError, AssertionError):
            tree = cleaner.clean_html(lxml.html.parse(str(p)))
            #for c in tree.xpath('//div/[@class="thread-content"]'):
            thread, *_ = tree.xpath('//section[@class="thread"]')
            title, *_ = thread.xpath('./h3[@class="post-title"]')
            date, author, difficulty, *_ =  thread.xpath('//dl/dd')
            title = title.text_content().strip()
            date = date.text_content().strip()
            author = author.text_content().strip()
            difficulty = difficulty.text_content().strip()
            #print([title, date, author, difficulty])
            sentences = list()
            tokenized = list()
            for c in thread.xpath('./div[@class="thread-content"]'):
                text = c.text_content().strip()
                sentence = [s for s in jieba.cut(text, HMM=True)]
                sentences.append(sentence)
            comments = list()
            for comment in tree.xpath('//section[@id="Reply"]//li[contains(@class,"comment")]'):
                #print(comment.text_content().strip())
                try:
                    #print([c.text_content().strip() for c in comment.xpath('div')])
                    reply, *_ = comment.xpath('./div/div/div[contains(@class,"comment-content")]')
                    reply_date, *_ = comment.xpath('./div/div/div/div/i[contains(@class,"fa-clock-o")]')
                    reply_author, *_ = comment.xpath('./div/div/div/div/a')
                    reply = reply.text_content().strip()
                    sentence = [s for s in jieba.cut(reply, HMM=True)]
                    sentences.append(sentence)
                    comments.append({
                        'content': reply,
                        'date': reply_date.get('title'),
                        'author': {
                            'href': reply_author.get('href'),
                            'title': reply_author.get('title'),
                        }
                    })
                except ValueError:
                    pass
                del comment
            processed = {
                '__VERSION__': __VERSION__,
                'title': title,
                'date': date,
                'author': author,
                'difficulty': difficulty,
                'sentences': sentences,
                'tokenized': None,
                'comments': comments,
            }
        if processed is not None:
            if processed['tokenized'] is None:
                # Newly processed document
                tokenized = list()
                for sentence in processed['sentences']:
                    Vocabulary.add_words(sentence)
                    sentence = Vocabulary.tokenize_sentence(sentence)
                    tokenized.append(sentence)
                processed['tokenized'] = tokenized
                with open(pkl_path, 'wb') as fp:
                    pickle.dump(processed, fp)
            else:
                # Document loaded from disk cache
                for sentence in processed['sentences']:
                    Vocabulary.add_words(sentence)
            Vocabulary.add_document(processed['tokenized'])
        '''for s in processed['tokenized']:
            for w in s:
                corpus.append(w)'''
        #corpus.append()
        count += 1
        if __LOG_LEVEL == logging.DEBUG:
            if count >= 8:
                break

    Vocabulary.save()

    exit()

    print(corpus)

    bigram_finder = BigramCollocationFinder.from_words(corpus)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 100)

    for bigram_tuple in bigrams:
        print(Vocabulary.untokenize(bigram_tuple))

    trigram_finder = TrigramCollocationFinder.from_words(corpus)
    trigrams = trigram_finder.nbest(TrigramAssocMeasures.chi_sq, 100)

    for trigram_tuple in trigrams:
        print(Vocabulary.untokenize(trigram_tuple))
