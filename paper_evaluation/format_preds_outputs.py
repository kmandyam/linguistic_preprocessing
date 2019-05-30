import sys

# converting pryzant format predictions into li et. al format predictions

# create a set of all the words in the original attribute vocab
# first argument is the paper output for formatting purposes
# second argument is the predictions file
predictions = []
with open(sys.argv[2]) as preds_file:
    for l in preds_file:
        predictions.append(l.strip())

with open(sys.argv[1]) as attr_file:
    i = 0
    for l in attr_file:
        split = l.strip().split("\t")
        print(split[0] + "\t" + predictions[i] + "\t" + split[2])
        i += 1
