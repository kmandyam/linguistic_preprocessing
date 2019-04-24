"""
args:
1 original vocab attribute file
2 our attribute vocab file
calculates the intersection of the two attribute vocabs
"""
import sys

their_set = set()
our_set = set()

# create a set of all the words in the original attribute vocab
for l in open(sys.argv[1]):
    their_set.add(l.strip())

# create a set of all the words in our attribute vocab
for l in open(sys.argv[2]):
    our_set.add(l.strip())

for tok in their_set.intersection(our_set):
    print(tok)



