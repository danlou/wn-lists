import codecs
import argparse
from string import punctuation
from nltk import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
english_sw = stopwords.words('english')

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger('lists')
logger.setLevel(logging.INFO)


"""
AUXILIARY
"""


def store_lines(l, path):
    with codecs.open(path, 'w', 'utf-8') as f:
        for e in l:
            f.write('%s\n' % e)


def get_all_hypernyms(synset):
    hypernyms = set()
    for hypernym in synset.hypernyms():
        hypernyms |= set(get_all_hypernyms(hypernym))
    return hypernyms | set(synset.hypernyms())


def get_all_hyponyms(synset):
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_all_hyponyms(hyponym))
    return hyponyms | set(synset.hyponyms())


def expand_synset(synset):
    return sorted(get_all_hyponyms(wn.wordnet.synset(synset)),
                  key=(lambda syn: len(get_all_hyponyms(syn))),
                  reverse=True)


def tokenize(text, lang='english'):

    def filter_stopwords(tokens, lang='english'):
        if lang is 'english':
            return [t for t in tokens if t not in english_sw]

    def filter_punctuation(tokens):
        return [t for t in tokens if t not in punctuation]

    tokens = word_tokenize(text)
    tokens = filter_punctuation(tokens)
    tokens = filter_stopwords(tokens, lang)
    return tokens


"""
METHODS TO CREATE DIFFERENT LISTS
"""


def list_part_of_speech(pos=['n', 'a', 'v', 'r']):

    def pos_lemmas(pos):
        lemmas = set()
        for noun_syn in wn.wordnet.all_synsets(pos):
            for lemma in noun_syn.lemmas():
                lemmas.add(lemma.name())
        return lemmas

    pos_name = dict(n='nouns', a='adjectives', v='verbs', r='adverbs')

    for p in pos:
        lemmas = pos_lemmas(p)
        store_lines(lemmas, 'wn-%s-%d.txt' % (pos_name[p], len(lemmas)))


def list_lemmas(root='entity.n.01', expanded=True):

    if expanded:
        syns = expand_synset(root)
    else:
        syns = [wn.wordnet.synset(root)]

    lines = []
    for syn in syns:
        lemmas = set(syn.lemma_names())
        hyponyms = get_all_hyponyms(syn)
        for hypo in hyponyms:
            lemmas |= set(hypo.lemma_names())

        logger.info('Retrieved %d hyponym lemmas from %s' % (len(lemmas), syn))
        lines += [': ' + syn.name()] + list(lemmas)

    store_lines(lines, 'wn-lemmas-%d.txt' % len(syns))


def list_definition_tokens(root='entity.n.01', expanded=True):

    if expanded:
        syns = expand_synset(root)
    else:
        syns = [wn.wordnet.synset(root)]

    lines = []
    for syn in syns:

        tokens = syn.lemma_names()
        tokens += tokenize(syn.definition())
        for hypo in get_all_hyponyms(syn):
            tokens += tokenize(hypo.definition())
        tokens = set(tokens)

        logger.info('Retrieved %d definition tokens from %s' % (len(tokens), syn))
        lines += [': ' + syn.name()] + list(tokens)

    store_lines(lines, 'wn-definition-tokens-%d.txt' % len(syns))


if __name__ == '__main__':
    # python wn-lists.py --list lemmas --root animal.n.01

    parser = argparse.ArgumentParser(description='Generates lists from WordNet')

    parser.add_argument('--list', required=True,
                        choices=['lemmas', 'tokens',
                                 'pos_n', 'pos_a', 'pos_v', 'pos_r'],
                        help='Create a list analysing one of these')

    parser.add_argument('--root', default='entity.n.01',
                        help='Starting synset')

    args = parser.parse_args()

    if args.list == 'lemmas':
        list_lemmas(args.root)

    elif args.list == 'tokens':
        list_definition_tokens(args.root)

    elif args.list == 'pos_n':
        list_part_of_speech('n')

    elif args.list == 'pos_a':
        list_part_of_speech('a')

    elif args.list == 'pos_v':
        list_part_of_speech('v')

    elif args.list == 'pos_r':
        list_part_of_speech('r')
