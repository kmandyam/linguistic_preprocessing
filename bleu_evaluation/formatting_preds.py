# the output of the delete_retrieve_generate repo includes a predictions file
# this file formats that predictions file for easier bleu score calculation
# usage: python formatting_preds.py /path/to/preds/file /path/to/human/references

import sys

preds = open(sys.argv[1]) # the preds file we scp'ed
human = open(sys.argv[2]) # the human file

output = open("formatted.preds", "w")
predictions = []
for line in preds:
    predictions.append(line.strip())

golds = []
for line in human:
    golds.append(line.strip().split("\t")[0])

for i in range(len(golds)):
    output.write(golds[i] + "\t" + predictions[i] + "\t" + "0" + "\n")

output.close()

