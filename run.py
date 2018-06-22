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
from keras.models import load_model

f_idx = 0
last_rects = []
vc = VideoCamera(416, 416)
out_wr = None

# YOU CAN MODIFY THESE
in_file = "8.avi"
is_fisheye = True
out_file = "output.avi"
show_yolo_detections = False
show_fisheye_decomp = False
use_4patch = True
stop_on_4patch_break = False
# END

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def preprocess_input(image, net_h, net_w):
    new_w = net_w
    new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255., (new_w, new_h), cv2.INTER_LANCZOS4)

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

def merge_fisheye_rects(ra):
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

            final_rects.append(((tx1 + tx2) / 2 / fix_w, (ty1 + ty2) / 2 / fix_h, (tx2 - tx1) / fix_w, (ty2 - ty1) / fix_h, 1, [1]))
    return final_rects

def test_frame(frame):
    global net
    global f_idx
    global last_rects
    global out_wr
    global color_table

    aspect = float(frame.shape[1]) / frame.shape[0]

    frame_r = cv2.resize(frame, (model.IMAGE_W, model.IMAGE_H))

    f_idx += 1

    if f_idx > 17:
        f_idx = 0
        if is_fisheye:
            fs = [0, 0, 0, 0]
            fs[0] = vc.get_frame(frame_r, 0)
            fs[1] = vc.get_frame(frame_r, 1)
            fs[2] = vc.get_frame(frame_r, 2)
            fs[3] = vc.get_frame(frame_r, 3)
            if show_fisheye_decomp:
                cv2.imshow('f1', fs[0][0])
                cv2.imshow('f2', fs[1][0])
                cv2.imshow('f3', fs[2][0])
                cv2.imshow('f4', fs[3][0])

            ra = []
            for f in fs:
                netout = net.predict(preprocess_input(f[0], model.IMAGE_H, model.IMAGE_W))
                ra.append((model.decode_netout(netout, 0.9, 0.4), f[0], f[1], f[2]))

            rects = merge_fisheye_rects(ra)
            last_rects = rects
        else:
            netout = net.predict(preprocess_input(frame, model.IMAGE_H, model.IMAGE_W))
            rects = model.decode_netout(netout, 0.9, 0.4)
            last_rects = rects
    else:
        rects = []

    frame = cv2.resize(frame, (int(340 * aspect), 340), interpolation = cv2.INTER_AREA)

    off_x = 0
    off_y = 0
    fix_w = frame.shape[1]
    fix_h = frame.shape[0]

    detections = []
    for r in rects:
        xmin  = int((r[0] - r[2]/2) * fix_w) - off_x
        xmax  = int((r[0] + r[2]/2) * fix_w) - off_x
        ymin  = int((r[1] - r[3]/2) * fix_h) - off_y
        ymax  = int((r[1] + r[3]/2) * fix_h) - off_y
        detections.append([xmin, ymin, xmax, ymax, model.bbox_score(r)])

    objs = tracker.update(np.array(detections) if detections else [], frame)

    for r in last_rects:
        if not show_yolo_detections:
            continue
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

    for d in objs:
        pts = np.array(d[5])
        cv2.polylines(frame, np.int32([pts]), 0, color_table[d[4] % len(color_table)], 2)
        for pt in d[5]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (255, 255, 0), 1)
        cv2.rectangle(frame, (int(d[0]),int(d[1])), (int(d[2]),int(d[3])), (0,255,0), 3)
        str1 = str(int(d[4]))
        if d[6]:
            str1 += "("
        for ii in range(len(d[6])):
            if ii != 0:
                str1 += ","
            str1 += str(d[6][ii])
        if d[6]:
            str1 += ")"
        cv2.putText(frame,
            str1,
            (int(d[0]), int(d[1]) - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            1e-3 * frame.shape[0] * 2,
            (0,255,0), 2)

    cv2.imshow('frame', frame)
    if out_wr == None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_wr = cv2.VideoWriter(out_file, fourcc, 25.0, (frame.shape[1], frame.shape[0]))
    out_wr.write(frame)
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
    global color_table

    color_table = []

    for i in range(0, 10):
        c = generate_new_color(color_table, pastel_factor = 0.5)
        color_table.append((int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))

    net = load_model('yolo3_model.h5')

    tracker = Sort(use_4patch, stop_on_4patch_break)

    test_vid(in_file)
