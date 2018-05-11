########################
## Extending yad2k for our purposes
########################

import os
import sys
import numpy as np

import tensorflow as tf
import cv2
import time

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

yad2kDir = os.path.abspath(os.path.join(os.getcwd() ,'..','..', 'yad2k_rep'))
if os.path.isdir(yad2kDir):

    sys.path.append(os.path.abspath(os.path.join(yad2kDir)))
    import yad2k_main
    import test_yolo
    from yad2k.models.keras_yolo import yolo_eval, yolo_head , yolo_boxes_to_corners

else:
    raise ImportError('couldnt import yad2k')

#some overrides for our purposes:
def yolo_filter_boxes_extend(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    all_scores = tf.boolean_mask(box_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes, all_scores


def yolo_eval_extend(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes, all_scores = yolo_filter_boxes_extend(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    all_scores = K.gather(all_scores, nms_index)
    return boxes, scores, classes, all_scores

#our extension to the detector class
class yad2kForBusDetection(object):
    def __init__(self,options):

        self.model_path = os.path.join('model_data','yolo.h5')
        self.classes_path = os.path.join('model_data','coco_classes.txt')
        self.anchors_path = os.path.join('model_data','yolo_anchors.txt')
        self.weights_path = 'yolov2.weights'
        self.cfg_path = 'yolov2.cfg'

        self.score_threshold = options['threshold']
        self.iou_threshold = options['iou_threshold']

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

        # Generate output tensor targets for filtered bounding boxes.
        self.yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
        # TODO: Wrap these backend operations with Keras layers.
        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes, self.all_scores = yolo_eval_extend(
            self.yolo_outputs,
            self.input_image_shape,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold)

        return yolo_model, sess , anchors , class_names

    def return_predict(self,image):

        boxes, scores, classes, all_scores = [self.boxes, self.scores, self.classes, self.all_scores]

        #image resize
        orig_height = image.shape[0]
        orig_width = image.shape[1]

        image_height = 608
        image_width = 608

        scale_height = image_height * 1. / orig_height
        scale_width = image_width * 1. / orig_width

        image_data = np.array(cv2.resize(image,(image_width,image_height),interpolation=cv2.INTER_CUBIC), dtype='float32') #get size from cfg file

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes, out_all_scores = self.sess.run(
            [boxes, scores, classes, all_scores],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image_height, image_width],
                K.learning_phase(): 0
            })
        print('Found {} boxes'.format(len(out_boxes)))

        #find 3 most segnificant classes
        classes_indexes = np.argsort(out_all_scores,axis=1)[:,-3:]
        classes_scores = np.sort(out_all_scores,axis=1)[:,-3:]
        classes_list = []
        for i in range(classes_indexes.shape[0]):
            classes_list.append(classes_indexes[i,(classes_scores[i,:] > self.score_threshold)])

        detections = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            predicted_class_list = []
            for class_idx in classes_list[i]:
                predicted_class_list.append(self.class_names[class_idx])

            top, left, bottom, right = box
            top = max(0, np.floor((top + 0.5)/scale_height).astype('int32'))
            left = max(0, np.floor((left + 0.5)/scale_width).astype('int32'))
            bottom = min(orig_height, np.floor((bottom + 0.5)/scale_height).astype('int32'))
            right = min(orig_width, np.floor((right + 0.5)/scale_width).astype('int32'))
            #print(label, (left, top), (right, bottom))

            box = {'label': predicted_class_list, 'confidence': score, 'topleft': {'x':left,'y':top},'bottomright':{'x':right,'y':bottom}}
            detections.append(box)


        return detections
