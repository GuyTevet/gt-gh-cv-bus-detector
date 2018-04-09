########################
## Extending darkflow for our purposes
########################

import os
import sys
import numpy as np

import tensorflow as tf
import time
import json
import numpy as np

darkFlowDir = os.path.abspath(os.path.join(os.getcwd(),'..' ,'..' , 'darkflow'))
if os.path.isdir(darkFlowDir):
    sys.path.append(darkFlowDir)
    from darkflow.net.framework import YOLOv2 , framework
    from darkflow.net.build import TFNet
    from darkflow.net.framework import create_framework
    from darkflow.dark.darknet import Darknet

    from darkflow.net import yolo
    from darkflow.net import yolov2
    from darkflow.net import vanilla

else:
    raise ImportError('couldnt import darkflow')

def processBoxForBusDetection(self, b, h, w, threshold):
    #just modifing the prediction function to give her 10 first gueses
    # instead of the first one only
    num_labels = min(3,b.probs[b.probs != 0].shape[0])
    if num_labels == 0:
        return None

    probs_copy = np.copy(b.probs)
    top_labels_idx = []

    for i in range(num_labels):
        top_idx = np.argmax(probs_copy)
        top_labels_idx.append(top_idx)
        probs_copy[top_idx] = 0. #in order to find the next index without sorting

    max_prob = b.probs[top_labels_idx[0]]
    labels = [self.meta['labels'][top_labels_idx[i]] for i in range(num_labels)]
    if max_prob > threshold:
        left = int((b.x - b.w / 2.) * w)
        right = int((b.x + b.w / 2.) * w)
        top = int((b.y - b.h / 2.) * h)
        bot = int((b.y + b.h / 2.) * h)
        if left < 0:  left = 0
        if right > w - 1: right = w - 1
        if top < 0:   top = 0
        if bot > h - 1:   bot = h - 1
        mess = ['{}'.format(label) for label in labels]
        return (left, right, top, bot, mess, top_labels_idx[0], max_prob)
    return None

class YOLOv2ForBusDetection(YOLOv2):
    #just modifing the prediction function to give her 10 first gueses
    # instead of the first one only
    process_box = processBoxForBusDetection



class darkflowForBusDetection(TFNet):

    def __init__(self, FLAGS, darknet=None):
        self.ntrain = 0

        if isinstance(FLAGS, dict):
            from darkflow.defaults import argHandler
            newFLAGS = argHandler()
            newFLAGS.setDefaults()
            newFLAGS.update(FLAGS)
            FLAGS = newFLAGS

        self.FLAGS = FLAGS
        if self.FLAGS.pbLoad and self.FLAGS.metaLoad:
            self.say('\nLoading from .pb and .meta')
            self.graph = tf.Graph()
            device_name = FLAGS.gpuName \
                if FLAGS.gpu > 0.0 else None
            with tf.device(device_name):
                with self.graph.as_default() as g:
                    self.build_from_pb()
            return

        if darknet is None:
            darknet = Darknet(FLAGS)
            self.ntrain = len(darknet.layers)

        self.darknet = darknet
        args = [darknet.meta, FLAGS]
        self.num_layer = len(darknet.layers)
        #self.framework = create_framework(*args)
        self.framework = YOLOv2ForBusDetection(*args)

        self.meta = darknet.meta

        self.say('\nBuilding net ...')
        start = time.time()
        self.graph = tf.Graph()
        device_name = FLAGS.gpuName \
            if FLAGS.gpu > 0.0 else None
        with tf.device(device_name):
            with self.graph.as_default() as g:
                self.build_forward()
                self.setup_meta_ops()
        self.say('Finished in {}s\n'.format(
            time.time() - start))






