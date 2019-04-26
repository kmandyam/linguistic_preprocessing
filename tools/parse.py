from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

archive = load_archive(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"
        )
predictor = Predictor.from_archive(archive, 'constituency-parser')

def parse_sentence(sentence):
    output = predictor.predict_json({"sentence": sentence})
    return output

# Procedure for deleting attribute markers based on constituency parse:
# take the sentence as an input
# make sure everything is lowercase and there's no punctuation
# retrieve the constituency parse
# parse['hierplane_tree']['root'] has all the good information
# for each level in the tree, find all non terminals
# for each level, construct all bigrams of nonterminals that don't exceed token length 4


# a function that chooses which spans to consider for scoring based on the parse
def retrieve_spans(parse):
    tree = parse['hierplane_tree']['root']
    words = tree['word'].strip().split()
    spans = set()
    if len(words) <= 4:
        spans.add(tree['word'].strip())
    retrieve_span_helper(tree, spans)
    return spans


# a function which recursively iterates over the tree to find spans
def retrieve_span_helper(node, spans):
    # if I don't have any children, then ignore
    if 'children' not in node:
        return
    else:
        # we do have children, so we need to find all combinations of them
        children_text = []
        for i in range(len(node['children'])):
            # check to see if we have any non terminals here
            # we don't want to consider children combinations of non terminals
            # if 'children' in node['children'][i]:
            children_text.append(node['children'][i]['word'])

        combinations = find_children_combinations(children_text)
        spans.update(combinations)
        for i in range(len(node['children'])):
            retrieve_span_helper(node['children'][i], spans)


# a function which finds all valid spans of children
def find_children_combinations(children):
    # enumerate all combos and don't add ones that exceed word limit
    combinations = set()
    # unigrams
    for child in children:
        if len(child.split()) <= 4:
            combinations.add(child)
    # bigrams
    for i in range(len(children) - 1):
        gram_a = children[i]
        gram_b = children[i+1]
        if len(gram_a.split()) + len(gram_b.split()) <= 4:
            combinations.add(gram_a + " " + gram_b)
    return combinations


# if __name__ == "__main__":
#     sentence = "a little dirty on the inside , but wonderful people that work there"
#     sentence = re.sub("[.]", "", sentence)
#     sentence = sentence.lower().strip()
#
#     parse = parse_sentence(sentence)
#     spans = build_tree(parse)
#     print(spans)

