'''
Created 2016 at NTNU.

This file defines the RunAll class which is used for generating the art.

This code is an expanded and modified version on the Deepdream code available at:
https://github.com/google/deepdream/blob/master/dream.ipynb
'''


import sys
sys.path.append("/usr/lib/python2.7/dist-packages/")
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import scipy as sp
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import time
import datetime

import glob
import getWord
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import os

from pathnames import *
import pprint

class RunAll:
    def make_step(self):
        '''Basic gradient ascent step.'''
        self.src = self.net.blobs['data'] # input image is stored in Net's 'data' blob
        self.dst = self.net.blobs[self.end]
        ox, oy = np.random.randint(-self.jitter, self.jitter+1, 2)
        self.src.data[0] = np.roll(np.roll(self.src.data[0], ox, -1), oy, -2) # apply jitter shift
        self.net.forward(end=self.end)


        # specify the optimization objective
        if "no guide" in self.word.lower():
            if self.focus != None:
                one_hot = np.zeros_like(self.dst.data)
                one_hot.flat[self.focus] = 1.
                self.dst.diff[:] = one_hot
            else:
                self.objective_L2()
        else:
            self.objective_guide()
        self.net.backward(start=self.end)
        g = self.src.diff[0]
        # apply normalized ascent step to the input image
        self.src.data[:] += self.step_size/np.abs(g).mean() * g
        self.src.data[0] = np.roll(np.roll(self.src.data[0], -ox, -1), -oy, -2) # unshift image
        if self.clip:
            bias = self.net.transformer.mean['data']
            self.src.data[:] = np.clip(self.src.data, -bias, 255-bias)

    def preprocess(self, net, img):
        return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
    def deprocess(self, net, img):
        return np.dstack((img + net.transformer.mean['data'])[::-1])
    def objective_L2(self):
        self.dst.diff[:] = self.dst.data

    def deepdream(self):
	    # prepare base images for all octaves
        self.base_img = self.img
        self.octaves = [self.preprocess(self.net, self.base_img)]
        for i in xrange(self.octave_n-1):
            self.octaves.append(nd.zoom(self.octaves[-1], (1, 1.0/self.octave_scale,1.0/self.octave_scale), order=1))

        self.src = self.net.blobs['data']
        self.detail = np.zeros_like(self.octaves[-1]) # allocate image for network-produced details

        for octave, octave_base in enumerate(self.octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
	            # upscale details from the previous octave
                h1, w1 = self.detail.shape[-2:]
                self.detail = nd.zoom(self.detail, (1, 1.0*h/h1,1.0*w/w1), order=1)
            self.src.reshape(1,3,h,w) # resize the network's input image size
            self.src.data[0] = octave_base+self.detail
            for i in xrange(self.iter_n):
                time.sleep(0.01)
                self.make_step()
                # visualization
                vis = self.deprocess(self.net, self.src.data[0])
                if not self.clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
	            clear_output()
	        # extract details produced on the current octave
            self.detail = self.src.data[0]-octave_base
	    # returning the resulting image
        return self.deprocess(self.net, self.src.data[0])

    def objective_guide(self):
        x = self.dst.data[0].copy()
        y = self.guide_features
        ch = x.shape[0]
        x = x.reshape(ch,-1)
        y = y.reshape(ch,-1)
        A = x.T.dot(y) # compute the matrix of dot-products with guide features
        self.dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

    def word_to_picture(self):
        self.img = np.float32(PIL.Image.open(self.input_image))
        self.result_dir = os.getcwd() + '/generate/' + self.dir_session + "/guided/"  + self.testname + "_" + self.name + "/"
        time.sleep(0.01)
        if self.database == 'ImageNet API':
            self.engine = 'IMAGENET'
            self.searchWord = self.word.replace(" ", "_")
        else:
            self.engine = 'GOOGLE'
            self.searchWord = self.word
        if len(glob.glob('words/' + self.word + '/*.jpg')) < self.num_images:
            print("Missing pictures for guiding. Please get pictures for the word: " + self.searchWord)
            return 0
        getWord.create_dir(self.result_dir)
        sp.misc.imsave(self.result_dir + str(0) + '.jpg', self.img)
        for i in range(1, self.num_images + 1):
            self.generateFromPicture('words/' + self.searchWord + '/' + str(i-1) + '.jpg')
            self.test = self.deepdream()
            sp.misc.imsave(self.result_dir + str(i) + '.jpg', self.test)
            self.img = np.float32(PIL.Image.open(self.result_dir + str(i) + '.jpg'))
            time.sleep(0.01)
            print( str(100 * (i) / (self.num_images)) + '% done')

    def generateFromPicture(self, filename):
        guide = np.float32(PIL.Image.open(filename))
        h, w = guide.shape[:2]
        self.src, self.dst = self.net.blobs['data'], self.net.blobs[self.end]
        self.src.reshape(1,3,h,w)
        self.src.data[0] = self.preprocess(self.net, guide)
        self.net.forward(end=self.end)
        self.guide_features = self.dst.data[0].copy()

    #Generate without any guide
    def generate_from_net(self):
        self.img = np.float32(PIL.Image.open(self.input_image))
        self.result_dir = os.getcwd() + '/generate/' + self.dir_session + "/noguide/" + self.testname + "_" + self.name + "/"
        getWord.create_dir(self.result_dir)
        sp.misc.imsave(self.result_dir + str(0) + '.jpg', self.img)
        for i in range(1, self.num_images + 1):
            self.img=self.deepdream()
            sp.misc.imsave(self.result_dir + str(i) + '.jpg', self.img)
            time.sleep(0.01)
            print( str(100 * (i) / (self.num_images)) + '% done')


    def __init__(self, test, input_image, word, num_images, database, iterations, octaves, octave_scale, model_name, layer, jitter, step_size, mean, focus, dir_session, full_run=False, net_params=["net_fn","param_fn","mean"]):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.testname = test
        self.input_image = input_image
        self.word = word.lower()
        self.num_images = num_images
        self.database = database
        self.iter_n = iterations
        self.octave_n = octaves
        self.octave_scale = octave_scale
        self.model_name = model_name
        self.layer = layer
        self.jitter = jitter
        self.step_size = step_size
        self.dir_session = dir_session
        self.end = layer
        self.guide_features = 0
        self.clip = True
        self.mean = mean #for testing gen with different mean
        self.focus = focus
        if model_name == "Cars":
            self.end = layer.replace("/","_")
        self.mod_par = {}
        if full_run:
            self.mod_par["net_fn"] = net_params[0]
            self.mod_par["param_fn"] = net_params[1]
            self.mod_par["mean"] = net_params[2]
        else:
            for param in ["net_fn","param_fn","mean"]:
                self.mod_par[param] = pathnames_list[model_name][param]
        self.model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(self.mod_par["net_fn"]).read(), self.model)
        self.model.force_backward = True
        open('tmp.prototxt', 'w').write(str(self.model))
        self.net = caffe.Classifier('tmp.prototxt', self.mod_par["param_fn"], mean = np.float32(self.mean), channel_swap = (2,1,0))
        self.name = input_image.split('/')[-1].split('.')[0] + "_" + word + "_" + model_name + "_" + str(iterations) + "_" + str(num_images) + "_" + str(octaves) + "_" + str(octave_scale) + "_" + str(layer).replace("/", "-") + "_" + str(jitter) + "_" + str(step_size)

        if "no guide" in word.lower():
            self.generate_from_net()
        else:
            self.word_to_picture()
