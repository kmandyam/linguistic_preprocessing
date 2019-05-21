import nltk
import sys

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

preds = open(sys.argv[1]) # the preds output from delete, retrieve, generate
human = open(sys.argv[2]) # the human output

references = []
hypotheses = []

for line in preds:
    hypotheses.append(line.strip())

for line in human:
    references.append([line.strip().split("\t")[1]])


# hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
# reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']

# BLEU_score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

# sentence_bleu = sentence_bleu([reference], hypothesis)
# corpus_bleu = corpus_bleu([[reference]], [hypothesis])


# print("sentence bleu", sentence_bleu)
# print("corpus bleu", corpus_bleu)

corpus_score = corpus_bleu(references, hypotheses, [0.25, 0.25, 0.25, 0.25])
print("corpus score", corpus_score)