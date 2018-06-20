#!/usr/bin/python3

from keras import backend as K
import cv2
import os
import model
from PIL import Image
import threading, os, re, random
from random import shuffle
import numpy as np
from datetime import datetime
import tensorflow as tf
from sort import Sort
from fisheye_camera import VideoCamera

f_idx = 0
last_rects = []

vc = VideoCamera(608, 608)

def merge_rects(ra, orig_frame):
    ii = 0
    final_rects = []
    for rr in ra:
        ii += 1
        rects = rr[0]
        frame = rr[1]
        mapx = rr[2]
        mapy = rr[3]
        for r in rects:
            fix_w = frame.shape[1]
            fix_h = frame.shape[0]
            xmin  = int((r[0] - r[2]/2) * fix_w)
            xmax  = int((r[0] + r[2]/2) * fix_w)
            ymin  = int((r[1] - r[3]/2) * fix_h)
            ymax  = int((r[1] + r[3]/2) * fix_h)

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax < 0:
                xmax = 0
            if ymax < 0:
                ymax = 0

            if xmin >= fix_w:
                xmin = fix_w - 1
            if ymin >= fix_h:
                ymin = fix_h - 1
            if xmax >= fix_w:
                xmax = fix_w - 1
            if ymax >= fix_h:
                ymax = fix_h - 1

            x1 = mapx[ymin, xmin]
            y1 = mapy[ymin, xmin]

            x2 = mapx[ymin, xmax]
            y2 = mapy[ymin, xmax]

            x3 = mapx[ymax, xmax]
            y3 = mapy[ymax, xmax]

            x4 = mapx[ymax, xmin]
            y4 = mapy[ymax, xmin]

            if (x1 < 0) or (y1 < 0) or (x2 < 0) or (y2 < 0) or (x3 < 0) or (y3 < 0) or (x4 < 0) or (y4 < 0):
                continue

            tx1 = int(np.min(np.array([x1, x2, x3, x4])))
            ty1 = int(np.min(np.array([y1, y2, y3, y4])))
            tx2 = int(np.max(np.array([x1, x2, x3, x4])))
            ty2 = int(np.max(np.array([y1, y2, y3, y4])))

            if ((tx2 - tx1) / fix_w > 0.5) or ((ty2 - ty1) / fix_h > 0.5):
                continue

            cv2.rectangle(orig_frame, (tx1,ty1), (tx2,ty2), (255,0,0), 3)

            final_rects.append(((tx1 + tx2) / 2 / fix_w, (ty1 + ty2) / 2 / fix_h, (tx2 - tx1) / fix_w, (ty2 - ty1) / fix_h, 1, [1]))
    #cv2.imshow('ff', orig_frame)
    return final_rects

def test_frame(frame):
    global net
    global dummy_array
    global f_idx
    global last_rects

    keep_aspect = False

    aspect = float(frame.shape[1]) / frame.shape[0]

    if keep_aspect:
        frame_r = model.resize_fit(frame, (model.IMAGE_W, model.IMAGE_H), cv2.INTER_LANCZOS4)
    else:
        frame_r = cv2.resize(frame, (model.IMAGE_W, model.IMAGE_H))

    #cv2.imshow('frame_r', frame_r)

    f_idx += 1

    if f_idx > 17:
        ff = frame_r
        fs = [0, 0, 0, 0]
        fs[0] = vc.get_frame(frame_r, 0)
        fs[1] = vc.get_frame(frame_r, 1)
        fs[2] = vc.get_frame(frame_r, 2)
        fs[3] = vc.get_frame(frame_r, 3)
        #cv2.imshow('f1', fs[0][0])
        #cv2.imshow('f2', fs[1][0])
        #cv2.imshow('f3', fs[2][0])
        #cv2.imshow('f4', fs[3][0])

        ra = []
        for f in fs:
            frame_r = f[0]
            frame_r = frame_r / 255.0
            frame_r = np.expand_dims(frame_r, 0)
            netout = net.predict([frame_r, dummy_array])[0]
            ra.append((model.decode_netout(netout, 0.5, 0.3), f[0], f[1], f[2]))

        rects = merge_rects(ra, ff)
        f_idx = 0
        last_rects = rects
    else:
        rects = []

    frame = cv2.resize(frame, (int(480 * aspect), 480), interpolation = cv2.INTER_AREA)

    off_x = 0
    off_y = 0
    fix_w = frame.shape[1]
    fix_h = frame.shape[0]

    if keep_aspect:
        tmp1 = max(frame.shape[1], frame.shape[0])
        rf = model.aspect_fit((tmp1, tmp1), (frame.shape[1], frame.shape[0]))
        off_x = rf[0]
        off_y = rf[1]
        fix_w = tmp1
        fix_h = tmp1

    detections = []
    for r in rects:
        xmin  = int((r[0] - r[2]/2) * fix_w) - off_x
        xmax  = int((r[0] + r[2]/2) * fix_w) - off_x
        ymin  = int((r[1] - r[3]/2) * fix_h) - off_y
        ymax  = int((r[1] + r[3]/2) * fix_h) - off_y
        detections.append([xmin, ymin, xmax, ymax, model.bbox_score(r)])

    trackers = tracker.update(np.array(detections) if detections else [], frame)

    for r in last_rects:
        #continue
        xmin  = int((r[0] - r[2]/2) * fix_w) - off_x
        xmax  = int((r[0] + r[2]/2) * fix_w) - off_x
        ymin  = int((r[1] - r[3]/2) * fix_h) - off_y
        ymax  = int((r[1] + r[3]/2) * fix_h) - off_y
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255,0,0), 3)
        cv2.putText(frame,
                    str(r[4]),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * frame.shape[0],
                    (255,0,0), 2)

    for d in trackers:
        cv2.rectangle(frame, (int(d[0]),int(d[1])), (int(d[2]),int(d[3])), (0,255,0), 3)
        cv2.putText(frame,
                    str(int(d[4])),
                    (int(d[0]), int(d[1]) - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * frame.shape[0],
                    (0,255,0), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xff
    if k == 32:
        k = cv2.waitKey() & 0xff
    if k == 27:
        exit()

def test_vid(fn):
    cap = cv2.VideoCapture(fn)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        test_frame(frame)

    cap.release()

if __name__ == "__main__":
    global net
    global dummy_array
    global dummy_mask

    net = model.yolo()

    weight_reader = model.WeightReader("yolov2.weights")

    weight_reader.reset()
    nb_conv = 23

    for i in range(1, nb_conv+1):
        conv_layer = net.get_layer('conv_' + str(i))

        if i < nb_conv:
            norm_layer = net.get_layer('norm_' + str(i))

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])

        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])
        conv_layer.trainable = False
        norm_layer.trainable = False

    dummy_array = np.zeros((1,1,1,1,model.TRUE_BOX_BUFFER,4))

    tracker = Sort(use_dlib=True)

    test_vid("8.avi")
