'''
Created 2016 at NTNU.

This file is currently under development.
The purpose of the sentenceTerms class is to find short sentences from the text to generate more specific search terms.
'''

# -*- coding: utf-8 -*-

import nltk

class sentenceTerms:
    def traverse(self, t):
        try:
            t.label()
        except AttributeError:
            return
        if t.label() == self.types:
            self.sentence_list.append(t)
        else:
            for child in t:
                self.traverse(child)

    def make_sentences(self, rawtext, types):
        self.rawtext = unicode(rawtext, errors='ignore')
        self.sentences = nltk.sent_tokenize(self.rawtext) # NLTK default sentence segmenter
        self.sentences = [nltk.word_tokenize(sent) for sent in self.sentences] # NLTK word tokenizer
        self.sentences = [nltk.pos_tag(sent) for sent in self.sentences] # NLTK POS tagger
        self.sentences = [self.cp.parse(sent) for sent in self.sentences]
        self.sentence_list = []
        self.types = types
        for sent in self.sentences:
            self.traverse(sent)

    def __init__(self):
        #Alle NN burde med i NP fordi mange NN vises som NNP
        self.grammar = r"""
                NP: {<DT|PP+$>*<JJ>*<NN|NNS>+}
                NJNP: {<DT|PP+$>*<JJ>+<NN|NNS>+}
                JNP:{<JJ>+<NN.*>+}
                NINP:{<NP|JNP><IN>+<NP|JNP>}
                VNP:{<VB.*>+<NN.*>+<VB.*>*}
        """

        self.cp = nltk.RegexpParser(self.grammar)
