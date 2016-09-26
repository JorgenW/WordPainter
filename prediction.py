'''
This file can be for image classification.
'''

import caffe
import numpy as np


#caffe_root = './caffe'


MODEL_FILE = '../caffe/models/helieverest/deploy.prototxt'
PRETRAINED = '../caffe/models/helieverest/HELIEVEREST_iter_200000.caffemodel'

caffe.set_mode_gpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
               mean = np.float32([119.0, 119.0, 119.0]),
               channel_swap=(2,1,0),
               raw_scale=255,
               image_dims=(256, 256))

def caffe_predict(path):
        input_image = caffe.io.load_image(path)
        print path
        print input_image
        prediction = net.predict([input_image])


        print prediction
        print "----------"

        print 'prediction shape:', prediction[0].shape
        print 'predicted class:', prediction[0].argmax()


        proba = prediction[0][prediction[0].argmax()]
        ind = prediction[0].argsort()[-5:][::-1] # top-5 predictions

        prob_list = []
        for a in prediction[0].argsort():
            prob_list.append((a, prediction[0][a]))

        return prediction[0].argmax(), proba, ind, prob_list

pred, proba, ind, prob_list = caffe_predict('words/lion/333.jpg')
print "Prediction: ", pred
print "Proba: ", proba
print "ind: ", ind
for a, b in prob_list[::-1]:
    print a, "\t", b
