'''
Created 2016 at NTNU.

This file can be used to download Wikipedia articles fast and easy.

If you get unicode error, run with python 3.
For Linux with python2 as default, fetch the package python3-pip,
then run pip3 install wikipedia.
'''

import wikipedia    # pip install wikipedia
import os

if not os.path.exists('text_corpus'):
    os.makedirs('text_corpus')
if not os.path.exists('text_corpus/wiki'):
    os.makedirs('text_corpus/wiki')

while 1:
    name = input("Name of article: ")
    try:
        page = wikipedia.page(name)
        print('Title: ' + page.title)
        text_file = open("text_corpus/wiki/" + page.title, 'w')
        for line in page.content.split('\n'):
            if any(word in line for word in ['== Notes ==', '=== References ===', '=== Sources ===']):
                break
            text_file.write(line + '\n')
        text_file.close()
        print("Article stored in trainingfiles/wiki/" + page.title)
    except wikipedia.exceptions.PageError:
        print('Page does not exist.')
        print('Showing alternatives: ')
        for search in wikipedia.search(name): print(search)
    except wikipedia.exceptions.DisambiguationError as e:
        print('DisambiguationError!')
        print(e)
