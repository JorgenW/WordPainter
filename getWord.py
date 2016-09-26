'''
Created 2016 at NTNU.

This file contains the functions for image-fetching.
'''
# -*- coding: utf-8 -*-
import urllib2
import urllib
import os
import sys
import PIL.Image
from googleapiclient.discovery import build
from flickrapi import FlickrAPI
import math
import glob
import shutil
import time
import ConfigParser
import multiprocessing
import imghdr
import ssl
from subprocess import call
from datetime import datetime, timedelta
from options import get_word_options

loop_counter = 0


def words_to_single_folder(word_list, engine, timeout_t, full_run=False):
    model_path = get_word_options["model_path"]
    #Sort by lowest word number
    word_list.sort(key=lambda tup: tup[1])
    wlist, nlist = [], []
    for (w,n) in word_list:
        wlist.append(w.lower())
        nlist.append(n)
    #Make dictionary of word_list
    word_dict = {}
    for (a, b) in word_list:
        word_dict[a] = b
    print("word_dict: "+str(word_dict)), '\n'
    #for singleEntry in word_list:
    while len(wlist) != 0:
        print(wlist,nlist), '\n'
        w, n = wlist.pop(0), nlist.pop(0)
        print("\n\n" +w+": "+str(n)), '\n'
        re = fetch_word(w, timeout_t, engine, n)
    print("word_dict: ", word_dict), '\n'
    print("The pictures will now be gathered in one folder for you.."), '\n'

    if not full_run:
        del_dir('trainingimages')
        model_path = "trainingimages"

    create_dir(model_path)
    del_dir(model_path + "/images")
    create_dir(model_path + "/images")
    foldernames = []
    for singleEntry in word_list: # sett for word, number in word_list:
        word, number = singleEntry
        folder = word.lower().replace(" ", "_")
        # create_dir(model_path + "/images/" + folder)
        move_usable_images('words/'+word.lower()+'/', model_path + "/images/" + folder + '/')
        call(["mogrify -resize \'256!x256!\' " + model_path + "/images/" + folder + "/*.jpg"], shell=True)
        foldernames.append(folder)
    print("The pictures are ready in the folder " + model_path + "/images/"), '\n'

    if full_run:
        prep_for_net_training(foldernames, model_path)
        print("Training will now begin..\n")
        # Start training
        call('cd ' + model_path + ' && ../../build/tools/caffe train -solver ./solver.prototxt -weights ./bvlc_googlenet.caffemodel', shell=True)
        # To restart training after a break use: ../../build/tools/caffe train -solver ./solver.prototxt -snapshot ./MYNET_iter_5000.solverstate
        # in terminal, where the snapshot is the last snapshot created.
        print("Training is done..\n\nWill now generate standard proceedure")
        caffemodels = glob.glob(model_path + "/*0.caffemodel")
        for i, x in enumerate(caffemodels): # Terrible but working solution
            import re
            nodigs = re.sub("\D", "", x)
            caffemodels[i] = int(nodigs)
        caffemodels.sort()
        for x in glob.glob(model_path + "/*0.caffemodel"):
            if caffemodels[-1] in x:
                param_fn = model_path + "/" + x
        wordpainter(full_run=True, net_params=[model_path+"/deploy.prototxt",param_fn,[119.0, 119.0, 119.0]], model_name=get_word_options["model_name"])

    print("The program will now force Exit."), '\n'
    sys.exit()


def prep_for_net_training(foldernames, model_path):

    text_file = open(model_path + "/train.txt", 'w')
    category_file = open(model_path + "/categories.py", 'w')
    category_file.write("categories = [\n")

    category = 0
    for folder in foldernames:
        category_file.write("\t" + folder)
        if category < len(foldernames)-1:
            category_file.write(",\n")
        filelist = glob.glob(model_path + '/images/' + folder + '/*.jpg')
        filelist.sort()
        for image_path in filelist:
            text_file.write(image_path[len(model_path)+1:] + " " + str(category) + "\n")
        category += 1
    category_file.write("\n]")
    text_file.close()
    category_file.close()
    shutil.copyfile(model_path + "/train.txt", model_path + "/val.txt")

    print("Copying network config files..")
    shutil.copyfile("solver.prototxt", model_path + "/solver.prototxt")
    shutil.copyfile("train_val.prototxt", model_path + "/train_val.prototxt")
    shutil.copyfile("deploy.prototxt", model_path + "/deploy.prototxt")

    if not os.path.isfile("bvlc_googlenet.caffemodel"):
        print("Downloading GoogLeNet")
        import urllib
        urllib.urlretrieve("http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel", "bvlc_googlenet.caffemodel")
    shutil.copyfile("bvlc_googlenet.caffemodel", model_path + "/bvlc_googlenet.caffemodel")

    print("Updating train_val.prototxt")
    with open(model_path + '/train_val.prototxt', 'r') as file:
        data = file.readlines()
    data[916] = "    num_output: " + str(len(foldernames)) + "\n"
    data[1679] = "    num_output: " + str(len(foldernames)) + "\n"
    data[2392] = "    num_output: " + str(len(foldernames)) + "\n"
    with open(model_path + '/train_val.prototxt', 'w') as file:
        file.writelines( data )

    print("Updating deploy.prototxt")
    with open(model_path + '/deploy.prototxt', 'r') as file:
        data = file.readlines()
    data[2140] = "    num_output: " + str(len(foldernames)) + "\n"
    with open(model_path + '/deploy.prototxt', 'w') as file:
        file.writelines( data )
    print("Done preparing for training.")


def redist(nlist, leftover):
    if leftover >= 0:
        return nlist
    new_nlist = []
    for i, e in enumerate(nlist):
        new_nlist.append( e + int(abs(leftover) * e / sum(nlist)) )
    return new_nlist

def move_usable_images(from_folder, to_folder):
    image_number = 0
    print("Moving usable pictures from " + from_folder + " to " + to_folder), '\n'
    create_dir(to_folder)
    for i in glob.glob(from_folder+'*.jpg'):
        if imghdr.what(i) == 'jpeg':
            if os.path.getsize(i) > 500:
                f = open(i, "r")
                if len(f.read()) > 500:
                    shutil.copyfile(str(i), to_folder + str(image_number) + ".jpg")
                    image_number = image_number + 1
                f.close()

def create_dir(folder_path):
    print("Creating folder: " + folder_path), '\n'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def del_dir(folder_path):
    print("Cleaning: " + folder_path), '\n'
    try:
        shutil.rmtree(folder_path)
    except OSError, e:
        print(e), '\n'

#Currently not in use
def find_existing_pictures(folder_path):
    print("Looking for local pictures.."), '\n'
    number_of_images = 0
    namelist = []
    for i in glob.glob(folder_path + '*.jpg'):
        number_of_images += 1
        namelist.append(i.split('/')[-1])
    for i in glob.glob(folder_path+'*.png'):
        number_of_images += 1
        namelist.append(i.split('/')[-1])
    return namelist, len(namelist)

def prep_processes(html, n_to_get, folder_path):
    #Creates a list of lists with N_OF_PROCESSES processes
    url = ''
    fetch_list, fetch_processes = [], []
    count = 0
    for c in html:
        if count == n_to_get:
            break
        elif c == '\n':
            if len(fetch_processes) > get_word_options["N_OF_PROCESSES"]:
                fetch_list.append(fetch_processes)
                fetch_processes = []
            file_path = folder_path + str(count) + '.jpg'
            p = multiprocessing.Process(target = url_fetcher, args = (url, file_path, count), name = str(count))
            fetch_processes.append(p)
            url = ''
            count += 1
        else:
            url = url + c
    fetch_list.append(fetch_processes)
    return fetch_list

def start_processes(fetch_processes):
    for p in fetch_processes:
        p.start()

def terminate_processes(fetch_processes, folder_path):
    for p in fetch_processes:
        if p.is_alive():
            file_path = folder_path + p.name + '.jpg'
            p.terminate()
            try:
                os.remove(file_path)
                print("Unfinished file removed."), '\n'
            except Exception, e:
                print("The file could not be removed or did not exist: " + str(e.strerror)), '\n'

def google_search(q, n, start):
    Config = ConfigParser.ConfigParser()
    Config.read('config.txt')
    devkey = Config.get('API Keys', 'google')
    cx = Config.get('API Keys', 'google_cx')
    html = ''
    service = build("customsearch", "v1", developerKey=devkey)
    try:
        res = service.cse().list(
            q=q,
            cx=cx,
            searchType='image',
            #fileType = 'jpg',
            safe = "high",
            imgType = "photo",
            num=10,
            #safe='off',
            start=start ,
        ).execute()
    except urllib2.HTTPError:
        print("INDEX TOO HIGH"), '\n'
    for item in res['items']:
        html += item['link'] + '\n'
    return html

def read_glossary(word):
    line_count = 0
    wordnet_id = "-1"
    print("Looking for word in glossary.."), '\n'
    glossary = file('index.noun')

    #Find WordNet ID in glossary
    for line in glossary:
        split_line = line.split(' ')
        if word == split_line[0]:
            for temp in split_line: #bytt til split_line[1:]
                if len(temp) == 8 and (temp != split_line[0]):
                    print("Matching word found!")
                    return "n" + temp
        line_count += 1
    print("No word found in WordNet glossary! Next word.."), '\n'
    return '0'

def imagenet_get_url_list(word, c):
    word = word.replace(' ', '_')
    wordnet_id = read_glossary(word)
    if wordnet_id == '0':
        return '0'

    #Get url list from Imagenet
    print("Will be looking for WNID: " + wordnet_id), '\n'
    print("Fetching url list from ImageNet.."), '\n'
    html = '0'
    attempts = 0
    #Retry if no response from ImageNet
    while(html == '0' and attempts < get_word_options["IMAGENET_ATTEMPTS"]):
        attempts = attempts + 1
        try:
            response = urllib2.urlopen('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='+wordnet_id)
            html = response.read()
            response.close()
            print("Url list completed"), '\n'
        except IOError:
            print("ImageNet is not responding at the moment. Trying again in 10 seconds."), '\n'
            time.sleep(10)
        except KeyError:
            print("Content not found."), '\n'
        except urllib2.HTTPError, e:
            print("HTTP Error: " + str(e.code)), '\n'
        except urllib2.URLError, e:
            print("URL Error: " + str(e.reason)), '\n'
        except Exception:
            import traceback
            print("Generic exception: " + traceback.format_exc()), '\n'
    return html

def flickr_get_url_list(word, n_to_get):
    if n_to_get > get_word_options["SEARCH_LIMIT_FLICKR"]:
        n_to_get = get_word_options["SEARCH_LIMIT_FLICKR"]
    Config = ConfigParser.ConfigParser()
    Config.read('config.txt')
    flickr_public = Config.get('API Keys', 'flickr_public')
    flickr_secret = Config.get('API Keys', 'flickr_secret')
    extras='url_sq, url_t, url_s, url_q, url_m, url_n, url_z, url_c, url_l, url_o'
    from pprint import pprint
    flickr = FlickrAPI(flickr_public, flickr_secret, format='parsed-json')
    html, i, page, page_max = '', 0, 0, 1
    urls_i = ['url_m', 'url_c', 'url_n', 'url_o', 'url_sq'] # sorted by importance

    n_per_page = get_word_options["PAGE_LIMIT_FLICKR"] #math.ceil(n_to_get/10)
    pages = int(math.ceil(float(n_to_get)/float(n_per_page)))#int(n_to_get/n_per_page)
    print 'n_per_page: ', n_per_page
    nr_images_left = n_to_get
    for page in range(pages):
        # pip install requests[security] for https connections if error
        search = flickr.photos.search(text=word, per_page=n_per_page, page=page, extras=extras, sort='relevance')
        photos = search['photos']['photo']
        for part in photos: # extract urls from responce
            url = ''
            i = 0
            while url is '':
                try:
                    url = part[urls_i[i]] +'\n'
                    html += url
                except KeyError:
                    print(urls_i[i]+" not available, trying next"), '\n'
                    i = i + 1
                    if i == len(urls_i):
                        print("No acceptable url found."), '\n'
                        break
        print("Number of Urls: " + str(html.count('\n'))), '\n'
    return html

def fetch_word(word, timeout_t, engine, n_to_get):
    fetch_number = int(n_to_get * get_word_options["OVERSHOOT"])
    print("Fetching URLs..."), '\n'
    if engine.lower() == 'flickr':
        html = flickr_get_url_list(word, fetch_number)
    elif engine.lower() == 'google':
        html, i = '', 1
        while i <= fetch_number:
            html += google_search(word, 10, i)
            i = i + 10
    elif engine.lower() == 'imagenet':
        html = imagenet_get_url_list(word, fetch_number)
        if html == '0':
            print("Unable to retrieve url-list from ImageNet."), '\n'
            return -n_to_get
        elif 'The synset is not ready yet' in html:
            print("No pictures of this word on ImageNet"), '\n'
            return -n_to_get
    else:
        print("Search engine not supported. Exiting."), '\n'
        sys.exit()
    print("Number of Urls: " + str(html.count('\n'))), '\n'
    #Prepare fetching processes
    folder_path = 'words/flickr_temp/' + word + '/'
    del_dir(folder_path[:-1])
    create_dir(folder_path)
    fetch_list = prep_processes(html, fetch_number, folder_path)
    print("fetch_list length: "+str(len(fetch_list))), '\n'
    global loop_counter
    for fetch_processes in fetch_list:
        # The following sleep logic is to comply with data limits
        if engine.lower() == "flickr":
            if loop_counter * len(fetch_processes) > (get_word_options["SEARCH_LIMIT_FLICKR"]-get_word_options["N_OF_PROCESSES"]):
                print("Sleeping for an hour to comply with Flickr's data limit..\n\nWill begin again at " + format(datetime.now() + timedelta(hours=1), '%H:%M:%S'))
                time.sleep(3600)
                loop_counter = 0
        elif engine.lower() == "google":
            if loop_counter * len(fetch_processes) > 999:
                print("Sleeping for 24 hours to comply with Google's data limit..\n\nWill begin again at " + format(datetime.now() + timedelta(hours=24), '%H:%M:%S'))
                time.sleep(3600*24)
                loop_counter = 0
        print("Fetching pictures.."), '\n'
        start_processes(fetch_processes)
        loop_counter += 1
        time.sleep(timeout_t)
        print("Timout! Will now start to terminate processes."), '\n'
        # Termination
        terminate_processes(fetch_processes, folder_path)

    finished_folder = "words/" + word+"/"
    del_dir(finished_folder)
    create_dir(finished_folder)
    move_usable_images(folder_path, finished_folder)
    del_dir(folder_path[:-1])
    del_dir('words/flickr_temp')
    print("Fetching finished!\nYour images can be found in the folder words/" + word+"/"), '\n'
    return len(glob.glob('words/'+word+'/*.jpg'))-n_to_get

def url_fetcher(url, file_path, count):
    try:
        # Check if image is empty (above 6kb)
        a = urllib2.urlopen(url)

        if int(a.headers['Content-Length']) > 6000:
            # Check if image is saved locally already
            urllib.urlretrieve(url, file_path)
            if os.path.getsize(file_path) > 200:
                image = PIL.Image.open(file_path)   #No need to close
                # Resize picture max size: (250, 250)
                image.thumbnail((250, 250), PIL.Image.ANTIALIAS)
                image.save(file_path)
            else:
                os.remove(file_path)
                print("DELETED A PICTURE WITH NO SIZE: " + str(os.path.getsize(file_path)) + " Image number: " + str(count)), '\n'
        a.close()
    except IOError, e:
        print("Url is broken, trying next.. " + str(e) + "\n" + url), '\n'
    except KeyError, e:
        print("Content-Length not found, trying next.. " + str(e) + "\n" + url), '\n'
    except urllib2.HTTPError, e:
        print("HTTP Error: " + str(e.code) + "\n" + url), '\n'
    except urllib2.URLError, e:
        print("URL Error: " + str(e.reason) + "\n" + url), '\n'
    except Exception, e:
        import traceback
        print("Generic exception: " + traceback.format_exc()), '\n'
