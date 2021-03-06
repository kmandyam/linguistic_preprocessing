"""
python make_parse_attribute_vocab.py [corpus1 parses] [corpus2 parses] r

subsets a file by finding the words most associated with
one of two corpuses. threshold is r ( # in corpus_a  / # in corpus_b )
uses parse candidates from the compute_corpus_parses.py output

pass the array into the CountVectorizer (without the ngram argument)
the CountVectorizer takes care of calculating all salience scores, etc.
to calculate the attribute markers, we iterate through span candidates
  for both corpuses, and then choose the markers that have the
  highest salience scores
"""
import sys
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from tqdm import tqdm

class ParseSalienceCalculator(object):
    def __init__(self, pre_corpus, post_corpus, tokenize):

        self.vectorizer = CountVectorizer(tokenizer=tokenize)

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


def read_corpus_parses(filename):
    corpus_parses = []

    i = 0
    for line in open(filename):
        if i % 2 != 0:
            # we're at a list of parses
            parses = line.rstrip('\n').split(", ")
            parses = parses[:-1]
            parses = [parse for parse in parses if parse]
            corpus_parses.extend(parses)
        i += 1

    return corpus_parses


corpus1_parses = read_corpus_parses(sys.argv[1])
corpus2_parses = read_corpus_parses(sys.argv[2])

def tokenize(text):
    return [text]


# the salience
#  ratio
r = float(sys.argv[3])

sc = ParseSalienceCalculator(corpus1_parses, corpus2_parses, tokenize)

print("marker", "negative_salience", "positive_salience")
def calculate_attribute_markers(corpus):
    for parse_candidate in tqdm(corpus):
        negative_salience = sc.salience(parse_candidate, attribute="pre")
        positive_salience = sc.salience(parse_candidate, attribute="post")

        if max(negative_salience, positive_salience) > r:
            print(parse_candidate, negative_salience, positive_salience)


calculate_attribute_markers(corpus1_parses)
calculate_attribute_markers(corpus2_parses)
