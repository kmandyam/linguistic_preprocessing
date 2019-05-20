"""
python compute_corpus_parses.py [vocab] [corpus]

compute span candidates from parses for all sentences
in a corpus given the vocab
"""

# get a list of all sentences in the corpus
# unk words in the sentence that aren't in the vocabulary
# for each sentence in the corpus, get all span candidates as an array
    # this part is the equivalent of getting all the ngrams in a sentence

import sys
from parse import parse_sentence, retrieve_spans
import pickle
import os
from tqdm import tqdm

vocab = set([w.strip() for i, w in enumerate(open(sys.argv[1]))])

corpus_sentences = [
    l.strip().split()
    for l in open(sys.argv[2])
]

corpus_path, corpus_file = os.path.split(sys.argv[2])

def unk_corpus(sentences):
    corpus = []
    for line in sentences:
        # unk the sentence according to the vocab
        line = [
            w if w in vocab else '<unk>'
            for w in line
        ]
        corpus.append(' '.join(line))
    return corpus


unked_corpus = unk_corpus(corpus_sentences)

# what i really want is a dictionary from sentences to spans
# saved in a file that i can read
# this dictionary is a map from the original sentence in the corpus
# to the list of spans produced by parsing the unked sentence
span_dict = {}
for i, line in tqdm(enumerate(unked_corpus)):
    parse = parse_sentence(line)
    spans = retrieve_spans(parse)
    original_sentence = ' '.join(corpus_sentences[i])
    print(original_sentence)
    for span in spans:
        print(span, end=", ")
    print()