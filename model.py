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
import math
from imgaug import augmenters as iaa

LABELS = ['person']

IMAGE_H, IMAGE_W = 416, 416
ANCHORS          = [10,37, 17,71, 28,104, 28,50, 42,79, 45,148, 70,92, 77,181, 193,310]
TRUE_BOX_BUFFER  = 500
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

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def bbox_label(bbox):
    return np.argmax(bbox[5])

def bbox_score(bbox):
    return bbox[5][bbox_label(bbox)]

def do_nms(boxes, nms_threshold, obj_threshold):
    # suppress non-maximal boxes
    for c in range(1):
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

def decode_netout2(netout, obj_threshold, anchors):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for i in range(grid_h * grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if objectness <= obj_threshold:
                continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row, col, b, :4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / IMAGE_W  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / IMAGE_H  # unit: image height

            # last elements are class probabilities
            classes = netout[row, col, b, 5:]

            box = [x, y, w, h, objectness, classes]

            boxes.append(box)

    return boxes

def decode_netout(yolos, obj_threshold, nms_threshold):
    boxes = []

    for i in range(len(yolos)):
        boxes += decode_netout2(yolos[i][0], obj_threshold, ANCHORS[(2 - i) * 6:(3 - i) * 6])

    boxes = do_nms(boxes, nms_threshold, obj_threshold)

    boxes2 = []

    for b in boxes:
        if b[2] >= 0.05 and b[3] >= 0.05 and b[2] <= 0.5 and b[3] <= 0.8:
            boxes2.append(b)

    return boxes2
