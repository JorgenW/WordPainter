'''
Created 2016 at NTNU.

A complete run will start here and create search terms according to the options.py file.
'''


import getWord
import re
from text_processing.noun_extractor import *
import glob
import sys
from text_processing.tf_idf import tfidf_func
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing.sentence_analysis.sentences import *
from nltk.stem.wordnet import WordNetLemmatizer
from options import text_options
from getWord import *


def perform_text_analysis(text_options):
    with open(text_options["textfile_path"], 'r') as myfile:
        data = myfile.read().replace('\n', '')
    print("Finding nouns in text..")
    data_only_chars = re.sub("[^A-Za-z ]", " ", data)
    nouns = extract_nouns(data_only_chars, text_options["language"], text_options["proper_nouns"])
    non_stop_nouns = remove_stopwords(nouns, text_options["language"])

    '''Frequency'''
    if text_options["noun_extraction_method"] == "frequency":
        print("Extracting nouns by frequency..\n")

        if text_options["even_noun_distribution"]:
            result_noun_extraction = list_queries_from_text(data_only_chars, 1.0, text_options["TOTALIMAGES"], text_options["language"], text_options["proper_nouns"]) # Get nouns and number of images
            for i, (word, number) in enumerate(result_noun_extraction):
                result_noun_extraction[i] = (word, text_options["TOTALIMAGES"]/text_options["NROFCATEGORIES"])
                if i >= text_options["NROFCATEGORIES"]:
                    result_noun_extraction = result_noun_extraction[:text_options["NROFCATEGORIES"]]
                    break
        else:
            result_noun_extraction = list_queries_from_text(data_only_chars, text_options["cutoff_percentage"], text_options["TOTALIMAGES"], text_options["language"], text_options["proper_nouns"]) # Get nouns and number of images
        if text_options["correct_difference"]:
            correct_difference_searchlist(result_noun_extraction)

    '''TF-IDF'''
    if text_options["noun_extraction_method"] == "tfidf":
        print("Extracting nouns from tf-idf results..")
        tfs, tfidf, file_names = tfidf_func(text_options["CHAPTERFOLDER"]) # get tf-idf
        noun_dict = extract_nouns_from_tfidf(tfidf, tfs, non_stop_nouns)
        print("Sorting nouns..")
        nounlistsorted = sort_tuple_dict(noun_dict)
        result_noun_extraction = make_searchlist(nounlistsorted, even=text_options["even_noun_distribution"], correct_difference=text_options["correct_difference"])

    print("Making tagpattern tree..\n")

    print result_noun_extraction, sum_searchlist(result_noun_extraction)

    '''Sentences'''
    if text_options["noun_phrase"] == "sentence":
        sentences = make_tagpattern_tree(text_options["textfile_path"], text_options["TAGPATTERN"])
        '''Needed: Choosing sentence, insert into searchlist'''
        result_text_analysis = result_noun_extraction

    '''Collocations'''
    if text_options["noun_phrase"] == "collocation":
        flipped_result_list = flip_tup_list(result_noun_extraction)
        _, collocation_list = combine_collocations(text_options["TOTALIMAGES"], flipped_result_list, data_only_chars, text_options["language"], text_options["collocations_number"], text_options["collocations_window_size"])
        result_text_analysis = flip_tup_list(collocation_list)

    '''None'''
    if text_options["noun_phrase"] == "none":
        result_text_analysis = result_noun_extraction

    '''Full run?'''
    if text_options["only_text_analysis"]:
        exit()
    else:
        words_to_single_folder(result_text_analysis, 'Flickr', 20, full_run=True)


def make_searchlist(nounlistsorted, even=True, correct_difference=True):
    '''
    :param nounlistsorted:
    :param even: Makes the searchlist numbers evenly distributed rather than defined by tfidf.
    :param correct_difference: Correct difference caused by rounding of numbers.
    :return: A list of tuples of the form (word, images_to_be_retrieved_for_the_word)
    '''
    searchlist = []
    sumtfidf = 0
    for l in nounlistsorted[:text_options["NROFCATEGORIES"]]:
        (paper, word, tfidf)= l
        sumtfidf = sumtfidf + tfidf
    for l in nounlistsorted[:text_options["NROFCATEGORIES"]]:
        (paper, word, tfidf)= l
        if not even:
            searchlist.append((word, int(((tfidf/sumtfidf)*text_options["TOTALIMAGES"]))))
        else:
            searchlist.append((word, text_options["TOTALIMAGES"]/text_options["NROFCATEGORIES"]))
    if correct_difference:
        searchlist = correct_difference_searchlist(searchlist)
    return searchlist

def sum_searchlist(searchlist):
    total = 0
    for l in searchlist:
        (word,number) = l
        total += number
    return total

def flip_tup_list(tup_list):
    flipped_result_list = []
    for (x,y) in tup_list:
        flipped_result_list.append((y,x))
    return flipped_result_list

def correct_difference_searchlist(searchlist):
    (word,number) = searchlist[0]
    searchlist[0] = (word, number+(text_options["TOTALIMAGES"]-sum_searchlist(searchlist))) # Correct the difference, added to the element with highest number
    return searchlist

def make_tagpattern_tree(textfile_path, TAGPATTERN):
    rawtext = open(textfile_path).read()
    sentences = sentenceTerms()
    sentences.make_sentences(rawtext, TAGPATTERN) # make tagpattern tree
    return sentences

def extract_nouns_from_tfidf(tfidf, tfs, nouns):
    feature_names = tfidf.get_feature_names()
    noun_dict = {}
    rows, cols = tfs.nonzero()
    for document, word_number in enumerate(cols): # extract nouns only from tf-idf
        if feature_names[word_number] in nouns:
            noun_dict[(rows[document], feature_names[word_number])] = tfs[rows[document], word_number]
    return noun_dict

def sort_tuple_dict(noun_dict):
    nounlistsorted = []
    temp = 0.0
    tempw = ''
    found = False
    for document, word in noun_dict:
        found = False
        for index, (k, sortedword, value) in enumerate(nounlistsorted):
            if noun_dict[(document, word)] > value:
                nounlistsorted.insert(index,(document, word, noun_dict[(document, word)]))
                found = True
                break
        if not found:
            nounlistsorted.append((document, word, noun_dict[(document, word)]))
    return nounlistsorted
