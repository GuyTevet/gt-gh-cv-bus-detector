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
import cv2

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

yad2kDir = os.path.abspath(os.path.join(os.getcwd() ,'..','..', 'yad2k_rep'))
if os.path.isdir(yad2kDir):

    sys.path.append(os.path.abspath(os.path.join(yad2kDir)))
    import yad2k_main
    import test_yolo
    from yad2k.models.keras_yolo import yolo_eval, yolo_head

else:
    raise ImportError('couldnt import yad2k')


class yad2kForBusDetection(object):
    def __init__(self,options):

        self.model_path = os.path.join('model_data','yolo.h5')
        self.classes_path = os.path.join('model_data','coco_classes.txt')
        self.anchors_path = os.path.join('model_data','yolo_anchors.txt')
        self.weights_path = 'yolov2.weights'
        self.cfg_path = 'yolov2.cfg'

        self.score_threshold = options['threshold']
        self.iou_threshold = 0.8 #FIXME - TBD

        self.yolo_model, self.sess , self.anchors, self.class_names = self.load_model()

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

        # Verify model, anchors, and classes are compatible
        num_classes = len(class_names)
        num_anchors = len(anchors)
        # TODO: Assumes dim ordering is channel last
        model_output_channels = yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        # Check if model is fully convolutional, assuming channel last order.
        model_image_size = yolo_model.layers[0].input_shape[1:3]
        is_fixed_size = model_image_size != (None, None)

        return yolo_model, sess , anchors , class_names


    def return_predict(self,image):
        # Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        yolo_outputs = yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))
        input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            yolo_outputs,
            input_image_shape,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold)

        # for image_file in os.listdir(test_path):
        #     try:
        #         image_type = imghdr.what(os.path.join(test_path, image_file))
        #         if not image_type:
        #             continue
        #     except IsADirectoryError:
        #         continue
        #
        #     image = Image.open(os.path.join(test_path, image_file))
        # if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        #     resized_image = image.resize(
        #         tuple(reversed(model_image_size)), Image.BICUBIC)
        #     image_data = np.array(resized_image, dtype='float32')
        # else:
        #     # Due to skip connection + max pooling in YOLO_v2, inputs must have
        #     # width and height as multiples of 32.
        #     new_image_size = (image.width - (image.width % 32),
        #                       image.height - (image.height % 32))
        #     resized_image = image.resize(new_image_size, Image.BICUBIC)
        #     image_data = np.array(resized_image, dtype='float32')
        #     print(image_data.shape)

        #image resize

        orig_height = image.shape[0]
        orig_width = image.shape[1]

        image_height = 608
        image_width = 608

        scale_height = image_height * 1. / orig_height
        scale_width = image_width * 1. / orig_width

        image_data = np.array(cv2.resize(image,(image_height,image_width),interpolation=cv2.INTER_CUBIC), dtype='float32') #get size from cfg file

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [boxes, scores, classes],
            feed_dict={
                self.yolo_model.input: image_data,
                input_image_shape: [image_height, image_width],
                K.learning_phase(): 0
            })
        print('Found {} boxes'.format(len(out_boxes)))

        # font = ImageFont.truetype(
        #     font='font/FiraMono-Medium.otf',
        #     size=np.floor(3e-2 * image_height + 0.5).astype('int32'))
        # thickness = (image_width + image_height) // 300

        detections = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #
            # draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor((top + 0.5)/scale_height).astype('int32'))
            left = max(0, np.floor((left + 0.5)/scale_width).astype('int32'))
            bottom = min(orig_height, np.floor((bottom + 0.5)/scale_height).astype('int32'))
            right = min(orig_width, np.floor((right + 0.5)/scale_width).astype('int32'))
            #print(label, (left, top), (right, bottom))

            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])

            box = {'label': [predicted_class], 'confidence': score, 'topleft': {'x':left,'y':top},'bottomright':{'x':right,'y':bottom}}
            detections.append(box)

        return detections
