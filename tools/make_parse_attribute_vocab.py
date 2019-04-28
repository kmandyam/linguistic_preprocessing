"""
python make_ngram_attribute_vocab.py [vocab] [corpus1] [corpus2] r

subsets a [vocab] file by finding the words most associated with
one of two corpuses. threshold is r ( # in corpus_a  / # in corpus_b )
uses ngrams
"""
import sys
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk import ngrams

class SalienceCalculator(object):
    def __init__(self, pre_corpus, post_corpus):
        self.vectorizer = CountVectorizer()

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

corpus1_parse_candidates = [
    l.strip().split()
    for l in open(sys.argv[2])
]

corpus2_parse_candidates = [
    l.strip().split()
    for l in open(sys.argv[3])
]

# the salience ratio
r = float(sys.argv[4])

# don't need to UNK because we do that before computing the
# parses anyways

sc = SalienceCalculator(corpus1_parse_candidates, corpus2_parse_candidates)


def calculate_attribute_markers(corpus):
    for parse_candidate in corpus:
        negative_salience = sc.salience(parse_candidate, attribute="pre")
        positive_salience = sc.salience(parse_candidate, attribute="post")

        if max(negative_salience, positive_salience) > r:
            print(parse_candidate)


calculate_attribute_markers(corpus1_parse_candidates)
calculate_attribute_markers(corpus2_parse_candidates)