import glob

'''
Created 2016 at NTNU.

This file contains the options for wordpainter.
'''

'''
Options for the text analysis is set in the text_options dictionary.

:option "noun_extraction_method": "frequency", "tfidf" :
:option "even_noun_distribution": Bool : True makes equal number of images per search term.
:option "noun_phrase": "none", "sentence", "collocation" : Select the sentence structure for search term.
:option "proper_nouns": Bool : Should proper nouns be included.
:option "language": "english", * : See NLTK for supported languages.
:option "cutoff_percentage": float :
:option "textfile_path": Full pathname to textfile, only necessary if tfidf is selected.
:option "CHAPTERFOLDER": Folder for corpus, only necessary if tfidf is selected.
:option "collocations_number": int : Top number of occurring collocations to evaluate.
:option "collocations_window_size": int : Number of words in collocation.
:option "TAGPATTERN": 'NP', 'NJNP', 'JNP', 'NINP', 'VNP' : Chose sentence structure, only necessary if sentence is chosen for noun_phrase.
:option "NROFCATEGORIES": int : Number of nouns to choose.
:option "TOTALIMAGES": int : Total number of images to fetch.
:option "only_text_analysis": Bool : Set this to true if you want to check the text analysis results before proceeding.
'''

text_options = {
    "noun_extraction_method":   "tfidf", # "frequency" or "tfidf"
    "even_noun_distribution":   True,
    "noun_phrase":              "collocation", # "none", "sentence" or "collocation"
    "proper_nouns":             False,
    "language":                 "english",

    "cutoff_percentage":        0.25,

    "textfile_path":            "./text_corpus/wiki/Dog",
    "CHAPTERFOLDER":            'text_corpus/wiki/',

    "collocations_number":      500,
    "collocations_window_size": 2,

    "TAGPATTERN":               'JNP', # Options: NP, NJNP, JNP, NINP, VNP

    "NROFCATEGORIES":           20,
    "TOTALIMAGES":              4000,
    "correct_difference":       True,

    "only_text_analysis":       False
}
text_options["CHAPTERS"] = len(glob.glob(text_options["CHAPTERFOLDER"] + '*'))

'''
Options for the image fetching is set in the get_word_options dictionary.

:option "model_name": Name of new network
:option "OVERSHOOT": Multiplier for number of pictures to fetch (example: 1.2)
:option "N_OF_PROCESSES": Number of paralell fetching processes
:option "IMAGENET_ATTEMPTS": Number of attempts to get urls from ImageNet
:option "PAGE_LIMIT_FLICKR": Limit of urls per page. Maximum limit is 500
:option "SEARCH_LIMIT_FLICKR": Limit of total images fetched is 3600 per hour
:option "caffe_path": Path to Caffe folder
'''

get_word_options = {
    "model_name": "dogwikinet",   # Name of new network
    "OVERSHOOT": 1.0,             # Multiplier for number of pictures to fetch
    "N_OF_PROCESSES": 250,        # Number of paralell fetching processes
    "IMAGENET_ATTEMPTS": 10,      # Number of attempts to get urls from ImageNet
    "PAGE_LIMIT_FLICKR": 500,     # Limit of urls per page. Maximum limit is 500
    "SEARCH_LIMIT_FLICKR": 3500,  # Limit of total images fetched is 3600 per hour
    "caffe_path": "../caffe/"     # Path to Caffe folder
}
get_word_options["model_path"] = get_word_options["caffe_path"] + "models/" + get_word_options["model_name"]
