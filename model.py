#!/usr/bin/python3

from keras import models
from keras import layers
from keras import callbacks
from keras import optimizers
from keras.utils.vis_utils import plot_model
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

IMAGE_H, IMAGE_W = 608, 608
GRID_H,  GRID_W  = 19 , 19
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
TRUE_BOX_BUFFER  = 50
ALPHA = 0.1

def rect_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return ()
    return (x, y, w, h)

def rect_union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def rect_inflate(a, dist):
    return (a[0] - dist, a[1] - dist, a[2] + dist * 2, a[3] + dist * 2)

def rect_inflate2(a, dist1, dist2):
    return (a[0] - dist1, a[1] - dist2, a[2] + dist1 * 2, a[3] + dist2 * 2)

def rect_4patch(a, i):
    off_x = a[2] / 4
    off_y = a[3] / 4
    a = (a[0], a[1], a[2] - off_x, a[3] - off_y)
    if i == 0:
        return a
    elif i == 1:
        return (a[0] + off_x, a[1], a[2], a[3])
    elif i == 2:
        return (a[0], a[1] + off_y, a[2], a[3])
    elif i == 3:
        return (a[0] + off_x, a[1] + off_y, a[2], a[3])

def pt_dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def aspect_fit(where, what):
    aspect = float(what[0]) / what[1]
    if aspect >= 1.0:
        rect = [0.0, 0.0, float(where[0]), float(where[0]) / aspect]
        if rect[3] <= where[1]:
            rect[1] = float(where[1] - rect[3]) / 2.0
        else:
            rect[3] = float(where[1])
            rect[2] = float(where[1]) * aspect
            rect[0] = float(where[0] - rect[2]) / 2.0
    else:
        rect = [0.0, 0.0, float(where[1]) * aspect, float(where[1])]
        if rect[2] <= where[0]:
            rect[0] = float(where[0] - rect[2]) / 2.0
        else:
            rect[2] = float(where[0])
            rect[3] = float(where[0]) / aspect
            rect[1] = float(where[1] - rect[3]) / 2.0

    return (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))

def resize_fit(frame, desired_size, interpolation):
    r = aspect_fit(desired_size, (frame.shape[1], frame.shape[0]))

    frame = cv2.resize(frame, (r[2], r[3]), interpolation = interpolation)

    color = [0, 0, 0]
    return cv2.copyMakeBorder(frame, r[1], desired_size[1] - r[1] - r[3], r[0], desired_size[0] - r[0] - r[2], cv2.BORDER_CONSTANT, value=color)

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

def yolo():
    input_image = layers.Input(shape=(IMAGE_H, IMAGE_W, 3))
    true_boxes  = layers.Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

    # Layer 1
    x = layers.Conv2D(32, (3, 3), strides=(1, 1),
                        padding='same', name='conv_1', use_bias=False)(input_image)
    x = layers.BatchNormalization(name='norm_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    # Layer 2
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_2')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)


    # Layer 3
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_3')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 4
    x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_4')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 5
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_5')(x)
    x= layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_6')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)


    # Layer 7
    x = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
    x= layers.BatchNormalization(name='norm_7')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 8
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_8')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_9')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 10
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_10')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 11
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_11')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)


    # Layer 12
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_12')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 13
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_13')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)


    skip_connection = x

    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    # Layer 14
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_14')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 15
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_15')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 16
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_16')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 17
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_17')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 18
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_18')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 19
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_19')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 20
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_20')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)


    # Layer 21
    skip_connection = layers.Conv2D(64, (1, 1), strides=(1, 1),
                                padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = layers.BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(skip_connection)
    skip_connection = layers.Lambda(space_to_depth_x2)(skip_connection)

    x = layers.concatenate([skip_connection, x])

    # Layer 22
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22',
                     use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_22')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 23
    x = layers.Conv2D((4 + 1 + CLASS) * 5, (1,1), strides=(1,1), padding='same', name='conv_23')(x)
    output = layers.Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

    # small hack to allow true_boxes to be registered when Keras build the model
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = layers.Lambda(lambda args: args[0])([output, true_boxes])

    model = models.Model([input_image, true_boxes], output)


    return model

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    x1_min  = box1[0] - box1[2]/2
    x1_max  = box1[0] + box1[2]/2
    y1_min  = box1[1] - box1[3]/2
    y1_max  = box1[1] + box1[3]/2

    x2_min  = box2[0] - box2[2]/2
    x2_max  = box2[0] + box2[2]/2
    y2_min  = box2[1] - box2[3]/2
    y2_max  = box2[1] + box2[3]/2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1[2] * box1[3] + box2[2] * box2[3] - intersect

    return float(intersect) / union

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)

def bbox_label(bbox):
    return np.argmax(bbox[5])

def bbox_score(bbox):
    return bbox[5][bbox_label(bbox)]

def decode_netout(netout, obj_threshold, nms_threshold):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = ANCHORS[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = ANCHORS[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]

                    box = [x, y, w, h, confidence, classes]

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(CLASS):
        sorted_indices = list(reversed(np.argsort([box[5][c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i][5][c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j][5][c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if bbox_score(box) > obj_threshold]
    boxes = [box for box in boxes if bbox_label(box) == 0]

    return boxes
