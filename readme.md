# WordPainter

This project was born during the spring semester of 2016 at NTNU in Trondheim, Norway. The project covers the course Computer Creativity. The original idea was a product of cooperation, but the final product presented here has been almost completely rewritten and has many improvements over the spring project and also has some parts removed.

The project aimed to create creative pictures by analyzing input texts and using the Google DeepDream code. Much was achieved during the spring semester, but it was during the summer that the project reached its current state. The program utilizes NLTK for noun extraction and fetches images from the Internet relevant to the nouns in the text. You can then train a NN (neural network) using the images fetched, or download a NN online. When you have a NN you can use it to generate images either with a guided word or with no guide. The guided word can be chosen from the text analysis and require that you fetch images of that word.

# What to do

It is recommended to use a fresh installation of Ubuntu.

Install Caffe
  - Make sure your Caffe folder and your wordpainter folder are in the same directory.
  - See the Wiki for installation guides.
  - Make sure to add GPU and cuDNN support if you have the hardware.

Add your text files for analyzis in /text_corpus. makewikifiles.py can be used.

Get your personal API keys from Google and Flickr and add them to config.txt, see example file for syntax (for flickr and google searches).

Add the paths to your NNs in pathnames.py.

Install any dependencies required.

Remove .example from files that have it and modify them if wanted.
 - The prototxt files in this folder will be copied and used for training og your networks. Many layers have been given new names: 'outnew'. Make sure you use the correct names for the layers in the proceedure file.

Modify the options.py, proceedure.py and pathnames.py files as wanted.

Run run_generate_art.py to generate art, or run run_wordpainter.py for full run.
