"""
args:
1 first attribute marker file
2 second attribute marker file
calculates the intersection of the two attribute markers
"""
import sys

their_set = set()
our_set = set()

# create a set of all the words in the original attribute vocab
with open(sys.argv[1]) as attr_file:
    next(attr_file)  # skip the header line
    for l in attr_file:
        split = l.strip().split()
        # pre_salience = split[-2]
        # post_salience = split[-1]
        marker = ' '.join(split[:-2])
        their_set.add(marker)

with open(sys.argv[2]) as attr_file:
    next(attr_file)  # skip the header line
    # create a set of all the words in our attribute vocab
    for l in attr_file:
        split = l.strip().split()
        # pre_salience = split[-2]
        # post_salience = split[-1]
        marker = ' '.join(split[:-2])
        our_set.add(marker)

print("Marker Set 0 Size", len(their_set))
print("Marker Set 1 Size", len(our_set))
print("Intersection Size", len(their_set.intersection(our_set)))
# print()
# print("Commonly Selected Tokens")
# for tok in their_set.intersection(our_set):
#     print(tok)



