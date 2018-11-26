__author__ = 'simon.hughes'

from spacy.lang.en import English
from Decorators import memoize
from collections import defaultdict
import sys

class BinaryRelation(object):
    def __init__(self, head, relation, child):
        self.relation = relation
        self.head = head
        self.child = "None" if (child is None or child.strip() == "") else child

    def __repr__(self):
        return "[%s]%s -> %s" % (self.relation, self.head, self.child)

class Relation(object):

    def __init__(self, head, relation, children):
        self.relation = relation
        self.head = head
        self.children = children
        self.__binary_relns_ = None

    def __repr__(self):
        skids = ",".join(self.children)
        return "[%s]%s -> %s" % (self.relation, self.head, skids)

    def binary_relations(self):
        if self.__binary_relns_ is not None:
            return self.__binary_relns_
        rels = []
        if len(self.children) == 0:
            rels.append(BinaryRelation(self.head, self.relation, None))
        else:
            for ch in self.children:
                rels.append(BinaryRelation(self.head, self.relation, ch))
        self.__binary_relns_ = rels
        return rels

# Python 2->3.x fix - emulate removed unicode function
# https://stackoverflow.com/questions/6812031/how-to-make-unicode-string-with-python3
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = lambda s: str(s)

class Parser(object):

    def __init__(self):
        if sys.version_info[0] >= 3:
            raise Exception("Does not work in Python 3.x - issues with SpaCy code")

        self.nlp = English()

    def parse(self, tokens):
        stokens = unicode(" ".join(tokens))

        tokens = self.__tokenize_(stokens)
        children_for_head = defaultdict(set)
        for token in tokens:
            children_for_head[token.head.i].add(token.string.strip())

        relations = []
        for token in tokens:
            kids = children_for_head[token.i]
            relations.append(Relation(token.head.string, token.dep_, list(kids)))

        assert len(relations) == len(tokens), "There are a different number of tokens to relations"
        return relations

    def pos_tag(self, tokens):
        stokens = unicode(" ".join(tokens))
        tokens = self.__tokenize_(stokens)
        return list(map(lambda t: t.pos_, tokens))

    def pos_tag2(self, tokens):
        stokens = unicode(" ".join(tokens))
        tokens = self.__tokenize_(stokens)
        return list(map(lambda t: t.tag_, tokens))

    def brown_cluster(self, tokens):
        stokens = unicode(" ".join(tokens))
        tokens = self.__tokenize_(stokens)
        return list(map(lambda t: str(t.cluster), tokens))

    def dep_vector(self, tokens):
        stokens = unicode(" ".join(tokens))
        tokens = self.__tokenize_(stokens)
        # yields a list of (300,) dimensional numpy arrays
        return list(map(lambda t: t.repvec, tokens))

    @memoize
    def __tokenize_(self, sentence):
        return list(self.nlp(sentence, tag=True, parse=True))

if __name__ == "__main__":

    parser = Parser()
    split = "The increasing levels of carbon dioxide caused coral bleaching".split(" ")
    parsed = parser.parse(split)
    print(parsed)

    tags = parser.pos_tag(split)
    print(tags)