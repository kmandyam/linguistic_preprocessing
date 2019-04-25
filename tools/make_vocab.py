"""
args:
1 corpus file (tokenized)
2 K
prints K most frequent vocab items
"""
import sys
from collections import Counter

print('<unk>')
print('<pad>')
print('<s>')
print('</s>')

c = Counter()

# TODO: generate all n-grams up to 4
# TODO: add n-grams to the counter and pick the top several

# argv 1 is entire corpus file (source and target concatenated)
# create a counter of every possible token (word in the corpus)
for l in open(sys.argv[1]):
    for tok in l.strip().split():
        c[tok] += 1

# argv 2 is the vocab size, get the vocab size most common
# words in the counter, and print the word
# we pipe the output to a vocab.txt file
for tok, _ in c.most_common(int(sys.argv[2])):
    print(tok)



