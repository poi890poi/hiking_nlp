import os
from pathlib import Path
import logging
import pickle

import numpy as np
import jieba
import nltk
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures
import lxml
import lxml.html
from lxml.html.clean import Cleaner


__VERSION__ = '0.1.2'
__LOG_LEVEL = logging.WARN
ROOT_DIR = './dataset/keepon/articles'


class Vocabulary():

    __document_count = 0
    __word2index = dict()
    __index2word = np.ndarray((0,), dtype=object)
    __word_frequency = np.ndarray((0,), dtype=np.uint)
    __doc_frequency = np.ndarray((0,), dtype=np.uint)

    @classmethod
    def word2index(cls):
        return cls.__word2index
    
    @classmethod
    def index2word(cls):
        return cls.__index2word

    @classmethod
    def word_frequency(cls):
        return cls.__word_frequency

    @classmethod
    def add_words(cls, sentence):
        for word in sentence:
            try:
                cls.__word_frequency[cls.__word2index[word]] += 1
            except KeyError:
                windex = len(cls.__word_frequency)
                cls.__word2index[word] = windex
                cls.__word_frequency.resize((len(cls.__word_frequency) + 1,))
                cls.__index2word.resize((len(cls.__index2word) + 1,))
                cls.__index2word[-1] = word
                cls.__word_frequency[-1] = 1

    @classmethod
    def add_document(cls, document):
        unique_words = np.array(tuple(set(w for s in document for w in s)), dtype=np.uint)
        if unique_words.size > 0:
            #print(cls.__doc_frequency, unique_words)
            new_len = max(len(cls.__doc_frequency), int(np.max(unique_words) + 1))
            #print(len(cls.__doc_frequency), int(np.max(unique_words) + 1), new_len)
            cls.__doc_frequency.resize(new_len)
            for word_index in unique_words:
                cls.__doc_frequency[word_index] += 1
            cls.__document_count += 1

    @classmethod
    def tokenize_sentence(cls, sentence):
        sentence = np.array([cls.__word2index[word] for word in sentence], dtype=np.uint)
        return sentence

    @classmethod
    def untokenize(cls, sentence):
        sentence = np.array([cls.__index2word[index] for index in sentence], dtype=object)
        return sentence

    @classmethod
    def load(cls, filename):
        cls.__filename = filename
        try:
            with open('./vocabulary.pkl', 'rb') as fp:
                vocabulary = pickle.load(fp)
            assert vocabulary['__VERSION__'] == __VERSION__
            cls.__word2index = vocabulary['word2index']
            cls.__index2word = np.copy(vocabulary['index2word'])
            cls.__word_frequency = np.copy(vocabulary['word_frequency'])
            cls.__document_count = vocabulary['document_count']
            cls.__doc_frequency = np.copy(vocabulary['doc_frequency'])
            del vocabulary
        except (FileNotFoundError, KeyError, AssertionError):
            pass

    @classmethod
    def save(cls):
        vocabulary = {
            '__VERSION__': __VERSION__,
            'word2index': cls.__word2index,
            'index2word': cls.__index2word,
            'word_frequency': cls.__word_frequency,
            'document_count': cls.__document_count,
            'doc_frequency': cls.__doc_frequency,
        }
        with open(cls.__filename, 'wb') as fp:
            pickle.dump(vocabulary, fp)


if __name__ == '__main__':

    Vocabulary.load('./dataset/vocabulary.pkl')

    count = 0
    cleaner = Cleaner()
    cleaner.javascript = True # This is True because we want to activate the javascript filter
    cleaner.style = True      # This is True because we want to activate the styles & stylesheet filter

    corpus = list()

    for p in Path(ROOT_DIR).rglob('*.html'):
        dir_, filename = os.path.split(p)
        prefix, extension = os.path.splitext(filename)
        pkl_path = os.path.join(dir_, '.'.join((prefix, 'pkl')))
        try:
            with open(pkl_path, 'rb') as fp:
                processed = pickle.load(fp)
                assert processed['__VERSION__'] == __VERSION__
        except (FileNotFoundError, AssertionError):
            print(count, pkl_path)
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
                Vocabulary.add_words(sentence)
                sentence = Vocabulary.tokenize_sentence(sentence)
                tokenized.append(sentence)
            comments = list()
            for comment in tree.xpath('//section[@id="Reply"]//li[contains(@class,"comment")]'):
                #print(comment.text_content().strip())
                print('-' * 32)
                try:
                    #print([c.text_content().strip() for c in comment.xpath('div')])
                    reply, *_ = comment.xpath('./div/div/div[contains(@class,"comment-content")]')
                    reply_date, *_ = comment.xpath('./div/div/div/div/i[contains(@class,"fa-clock-o")]')
                    reply_author, *_ = comment.xpath('./div/div/div/div/a')
                    reply = reply.text_content().strip()
                    sentence = [s for s in jieba.cut(text, HMM=True)]
                    sentences.append(sentence)
                    Vocabulary.add_words(sentence)
                    sentence = Vocabulary.tokenize_sentence(sentence)
                    tokenized.append(sentence)
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
            Vocabulary.add_document(tokenized)
            processed = {
                '__VERSION__': __VERSION__,
                'title': title,
                'date': date,
                'author': author,
                'difficulty': difficulty,
                'sentences': sentences,
                'tokenized': tokenized,
                'comments': comments,
            }
            print(processed)
            with open(pkl_path, 'wb') as fp:
                pickle.dump(processed, fp)
        '''for s in processed['tokenized']:
            for w in s:
                corpus.append(w)'''
        #corpus.append()
        count += 1
        if __LOG_LEVEL == logging.DEBUG:
            if count >= 256:
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
