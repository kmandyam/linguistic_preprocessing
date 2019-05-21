import sys
from nltk.translate.bleu_score import corpus_bleu

preds = open(sys.argv[1])  # the preds output from delete, retrieve, generate
human = open(sys.argv[2])  # the human output

references = []
hypotheses = []

for line in preds:
    hypotheses.append(line.strip())

for line in human:
    references.append([line.strip().split("\t")[1]])

corpus_score = corpus_bleu(references, hypotheses, [0.25, 0.25, 0.25, 0.25])
print("corpus score", corpus_score)