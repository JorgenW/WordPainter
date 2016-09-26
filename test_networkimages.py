'''
Created 2016 at NTNU.

This file can be used to easily fetch images from categories of your own choice.
The third parameter in words_to_single_folder is the timeout time for each fetching batch in seconds.
'''

from getWord import *

word_list = [('cat', 50), ('dog', 50), ('horse', 50), ('mouse', 50), ('moose', 50)]

words_to_single_folder(word_list, 'Flickr', 8)
