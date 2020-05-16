from process import Vocabulary

Vocabulary.load('./dataset/vocabulary.pkl')
print(Vocabulary.word2index())