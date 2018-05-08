########################
## Extending yad2k for our purposes
########################

import os
import sys
import numpy as np

import tensorflow as tf
import time
import json
import numpy as np

import argparse
import colorsys
import imghdr
import random
import argparse

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

yad2kDir = os.path.abspath(os.path.join(os.getcwd() ,'..','..', 'yad2k_rep'))
if os.path.isdir(yad2kDir):

    sys.path.append(os.path.abspath(os.path.join(yad2kDir)))
    import yad2k_main
    import test_yolo

else:
    raise ImportError('couldnt import yad2k')


class yad2kForBusDetection(object):
    def __init__(self,options):

        self.model_path = os.path.join('model_data','yolo.h5')
        self.classes_path = os.path.join('model_data','coco_classes.txt')
        self.anchors_path = os.path.join('model_data','yolo_anchors.txt')
        self.weights_path = 'yolov2.weights'
        self.cfg_path = 'yolov2.cfg'

        self.load_model()

    def create_model(self):
        # Convert the Darknet YOLO_v2 model to a Keras model
        fake_argv = [self.cfg_path, self.weights_path, self.model_path]
        yad2k_main._main(yad2k_main.parser.parse_args(args=fake_argv))

    def load_model(self):

        if not os.path.isfile(self.model_path):
            self.create_model()

        sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]

        with open(self.anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)

        yolo_model = load_model(self.model_path)


    def return_predict(self,image):
        pass