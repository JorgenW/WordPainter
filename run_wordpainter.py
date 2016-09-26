'''
Created 2016 at NTNU.

Runs the complete program according to the options.py file.
The complete program will:
- Analyze your selected text
- Fetch images based on your result
- Generate a new folder for your new network and move all needed files there.
- Train your network
- Generate art based on standard_proceedure.py 
'''

from analyzer import *
perform_text_analysis(text_options)
