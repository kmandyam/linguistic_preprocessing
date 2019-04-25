"""
python make_attribute_vocab.py [vocab] [corpus1] [corpus2] r

subsets a [vocab] file by finding the words most associated with
one of two corpuses. threshold is r ( # in corpus_a  / # in corpus_b )
"""
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from nltk import ngrams
from tqdm import tqdm

class SalienceCalculator(object):
    def __init__(self, pre_corpus, post_corpus):
        self.vectorizer = CountVectorizer(ngram_range=(1, 4))

        pre_count_matrix = self.vectorizer.fit_transform(pre_corpus)
        self.pre_vocab = self.vectorizer.vocabulary_
        self.pre_counts = np.sum(pre_count_matrix, axis=0)
        self.pre_counts = np.squeeze(np.asarray(self.pre_counts))

        post_count_matrix = self.vectorizer.fit_transform(post_corpus)
        self.post_vocab = self.vectorizer.vocabulary_
        self.post_counts = np.sum(post_count_matrix, axis=0)
        self.post_counts = np.squeeze(np.asarray(self.post_counts))

    def salience(self, feature, attribute='pre', lmbda=0.5):
        assert attribute in ['pre', 'post']

        if feature not in self.pre_vocab:
            pre_count = 0.0
        else:
            pre_count = self.pre_counts[self.pre_vocab[feature]]

        if feature not in self.post_vocab:
            post_count = 0.0
        else:
            post_count = self.post_counts[self.post_vocab[feature]]

        if attribute == 'pre':
            return (pre_count + lmbda) / (post_count + lmbda)
        else:
            return (post_count + lmbda) / (pre_count + lmbda)

# create a set of all words in the vocab
vocab = set([w.strip() for i, w in enumerate(open(sys.argv[1]))])

# get all the words in the source corpus
# if it's not in the vocab file, we unk the word
# produces an array
# corpus1 = sys.argv[2]
# corpus1 = [
#     w if w in vocab else '<unk>'
#     for l in open(corpus1)
#     for w in l.strip().split()
# ]

corpus1_sentences = [
    l.strip().split()
    for l in open(sys.argv[2])
]

# get all the words in the target corpus
# if it's not in the vocab file, we unk the word
# produces an array
# corpus2 = sys.argv[3]
# corpus2 = [
#     w if w in vocab else '<unk>'
#     for l in open(corpus2)
#     for w in l.strip().split()
# ]

corpus2_sentences = [
    l.strip().split()
    for l in open(sys.argv[3])
]

# the salience ratio
r = float(sys.argv[4])

def unk_corpus(sentences):
    corpus = []
    for line in sentences:
        # unk the sentence according to the vocab
        line = [
            w if w in vocab else '<unk>'
            for w in line
        ]
        # for i in range(1, 5):
        #     i_grams = ngrams(line, i)
        #     joined = [
        #         " ".join(gram)
        #         for gram in i_grams
        #     ]
        #     corpus.extend(joined)
        corpus.append(' '.join(line))
    return corpus


corpus1 = unk_corpus(corpus1_sentences)
corpus2 = unk_corpus(corpus2_sentences)

sc = SalienceCalculator(corpus1, corpus2)


def calculate_attribute_markers(corpus):
    for sentence in corpus:
        for i in range(1, 5):
            i_grams = ngrams(sentence.split(), i)
            joined = [
                " ".join(gram)
                for gram in i_grams
            ]
            for gram in joined:
                negative_salience = sc.salience(gram, attribute='pre')
                positive_salience = sc.salience(gram, attribute='post')
                if max(negative_salience, positive_salience) > r:
                    # print(gram, negative_salience, positive_salience)
                    print(gram)


calculate_attribute_markers(corpus1)
calculate_attribute_markers(corpus2)


# for each token in the vocab, if the salience with respect to the attribute is higher than
# a certain threshold, then we keep the token in the attribute vocab, otherwise we
# don't. Pipe the output to an attribute_vocab.txt

# for tok in vocab:
#     # print(tok, sc.salience(tok))
#     negative_salience = sc.salience(tok, attribute='pre')
#     positive_salience = sc.salience(tok, attribute='post')
#     # print(tok, negative_salience, positive_salience)
#     # if max(sc.salience(tok, attribute='pre'), sc.salience(tok, attribute='post')) > r:
#     #     print(tok)
#     if max(negative_salience, positive_salience) > r:
#         print(tok, negative_salience, positive_salience)



# TODO: this seems to be doing all the delete module, but without the n-gram


