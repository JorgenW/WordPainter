'''
Created 2016 at NTNU.

This file is responsible for the tf-idf calculations.
'''

import nltk
import string
import os
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
#from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
'''
#stemmer = PorterStemmer()
#stemmer = SnowballStemmer("english")
'''
stemmer = WordNetLemmatizer()


def stem_tokens(tokens):
    stemmed = []
    stemmer = WordNetLemmatizer()
    for item in tokens:
        stemmed.append(stemmer.lemmatize(item))#stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems

def tfidf_func(path):
    token_dict = {}
    file_names = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            file_path = subdir + os.path.sep + file
            f = open(file_path, 'r')
            text = f.read()
            lowers = text.lower()
            no_punctuation = lowers.translate(None, string.punctuation)
            token_dict[file] = no_punctuation.replace('\r\n', ' ')
    dictvalues = []
    for file in token_dict:
        dictvalues.append(token_dict[file])
        file_names.append(file)
    print("Calculating tf-idf.. (This can take some time)")
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(dictvalues)

    feature_names = tfidf.get_feature_names()
    termlist = []
    rows, cols = tfs.nonzero()

    print("Making list of all tf-idf...")
    for i in range(len(rows)):
        #if rows[i] == 1: # only chapter 2
            termlist.append((feature_names[cols[i]], rows[i], tfs[rows[i], cols[i]]))

    print("Sorting list...")
    termlist.sort(key=lambda tup: tup[2])

    # print("Write 30 highest tf-idf: ")
    # for g in range(30):
    #     print termlist[-g-1]

    #feature_names = tfidf.get_feature_names()
    #for col in tfs.nonzero()[1]:
    #    print feature_names[col], ' - ', tfs[0, col]

    return tfs, tfidf, file_names
