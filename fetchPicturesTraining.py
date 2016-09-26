'''
Created 2016 at NTNU.

Run this script to fetch pictures from a noun frequency analalyzis.
!OUTDATED!
'''

import getWord
import re
from text_processing.noun_extractor import *
import glob
import sys

reload(sys)
sys.setdefaultencoding('utf8')


# get textfile
textlist = []
print("These are the text files currently available for analysis: ")
for file in glob.glob('trainingfiles/*'):
    textlist.append( file.split('/')[-1] )
print(textlist)
textfile = raw_input("Name of textfile: ")
with open("./trainingfiles/" + textfile, 'r') as myfile:
    data = myfile.read().replace('\n', '')
data = re.sub("[^A-Za-z ]", " ", data) # Remove all but text
print("Textfile loaded.")

cutoff = float(raw_input("What cutoff cutoff percentage do you want?: %"))/100
n_of_images_to_be_retrieved = int( raw_input("Number of images to be retrieved?: ") )
language = raw_input("What language is the text: ").lower()
proper = bool( (raw_input("Are proper nouns allowed? (True/False): ")).lower() == ('true' or 't'))
if proper:
    print("Proper nouns will be included.")
else:
    print("Proper nouns will not be included.")

result_text_process = list_queries_from_text(data, cutoff, n_of_images_to_be_retrieved, language, proper)
print(result_text_process)
engine = raw_input("What image engine to use? (GOOGLE/IMAGENET/FLICKR): ")
timeout_t = int( raw_input("Timeout for threads in seconds: ") )
getWord.words_to_single_folder(result_text_process, engine, timeout_t)
