"""
@author: Mahmoud I.Zidan
"""

import cv2
import numpy as np
import model
from kalman_tracker import KalmanBoxTracker

'''Appearance Model'''
class KCFTracker:

  count = 0
  def __init__(self,bbox,img):
    #self.kalman_tracker = KalmanBoxTracker(bbox)
    self.tracker = cv2.TrackerKCF_create()
    self.last_pos = (int(bbox[0]),int(bbox[1]),int(bbox[2] - bbox[0]),int(bbox[3] - bbox[1]))
    r = model.rect_inflate2(self.last_pos, -self.last_pos[2] / 4, -self.last_pos[3] / 4);
    self.tracker.init(img, r)
    self.tracker.update(img)
    self.confidence = 0. # measures how confident the tracker is! (a.k.a. correlation score)

    self.trackers = [cv2.TrackerKCF_create(), cv2.TrackerKCF_create(), cv2.TrackerKCF_create(), cv2.TrackerKCF_create()]
    self.lp = [0, 0, 0, 0]
    self.tot_dist = [0.0, 0.0, 0.0, 0.0]
    self.bboxes = [0, 0, 0, 0]
    for i in range(4):
        r = model.rect_4patch(self.last_pos, i)
        self.trackers[i].init(img, r)
        self.lp[i] = (r[0] + r[2] / 2, r[1] + r[3] / 2)
        self.bboxes[i] = r

    self.time_since_update = 0
    self.id = KCFTracker.count
    KCFTracker.count += 1
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.lost = False
    self.points = []

  def predict(self,img):
    ok, bbox = self.tracker.update(img)
    #kbbox = self.kalman_tracker.predict(img)
    if ok:
        self.lost = False
        self.last_pos = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
        self.last_pos = model.rect_inflate2(self.last_pos, self.last_pos[2] / 2, self.last_pos[3] / 2)
        #self.kalman_tracker.update((int(bbox[0]),int(bbox[1]),int(bbox[2] + bbox[0]),int(bbox[3] + bbox[1])), img)
    else:
        self.lost = True
        #self.last_pos = (int(kbbox[0]),int(kbbox[1]),int(kbbox[2] - kbbox[0]),int(kbbox[3] - kbbox[1]))
        #self.tracker = cv2.TrackerKCF_create()
        #self.tracker.init(img, self.last_pos)
        #self.tracker.update(img)

    frame = img.copy()
    for i in range(4):
        ok, bbox = self.trackers[i].update(img)
        if ok:
            self.bboxes[i] = bbox
            ps = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            diff = model.pt_dist(ps, self.lp[i])
            self.lp[i] = ps
            self.tot_dist[i] += diff
            cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[0] + bbox[2]),int(bbox[1] + bbox[3])), (255,0,0), 3)

    idx = list(np.argsort(self.tot_dist))

    aaa = False

    for i in range(2):
        cx = 0.0
        cy = 0.0
        rct = None
        for j in idx:
            cx += self.lp[j][0]
            cy += self.lp[j][1]
            if rct is None:
                rct = self.bboxes[j]
            else:
                rct = model.rect_union(rct, self.bboxes[j])
        cx /= len(idx)
        cy /= len(idx)
        rct_good_w = self.bboxes[0][2] * 4 / 3
        rct_good_h = self.bboxes[0][3] * 4 / 3
        rct_good = (cx - rct_good_w / 2, cy - rct_good_h / 2, rct_good_w, rct_good_h)
        factor = (rct[2] * rct[3]) / (rct_good[2] * rct_good[3])
        if (factor <= 1.25):
            self.last_pos = rct_good
            self.points.append((cx, cy))
            if len(self.points) > 100:
                del self.points[0]
            self.lost = False
            if i != 0:
                for k in range(4):
                    self.trackers[k] = cv2.TrackerKCF_create()
                    r = model.rect_4patch(rct_good, k)
                    self.trackers[k].init(img, r)
                    self.lp[k] = (r[0] + r[2] / 2, r[1] + r[3] / 2)
                    self.tot_dist[k] = 0.0
                    self.bboxes[k] = r
                cv2.rectangle(frame, (int(rct_good[0]),int(rct_good[1])), (int(rct_good[0] + rct_good[2]),int(rct_good[1] + rct_good[3])), (0,0,255), 2)
                aaa = False
            break
        if i == 1:
            break
        del idx[0]
        del idx[0]
        del idx[0]

    cv2.imshow('frame2', frame)
    if aaa:
        cv2.waitKey()

    self.age += 1
    if (self.time_since_update > 0):
      self.hit_streak = 0
    self.time_since_update += 1

    return self.get_state()

  def update(self,bbox,img):
    #self.kalman_tracker.update(bbox, img)
    if bbox != []:
     self.time_since_update = 0
     self.hits += 1
     self.hit_streak += 1

    '''re-start the tracker with detected positions (it detector was active)'''
    if bbox != []:
      self.tracker = cv2.TrackerKCF_create()
      self.last_pos = (int(bbox[0]),int(bbox[1]),int(bbox[2] - bbox[0]),int(bbox[3] - bbox[1]))
      r = model.rect_inflate2(self.last_pos, -self.last_pos[2] / 4, -self.last_pos[3] / 4);
      self.tracker.init(img, r)
      self.tracker.update(img)
      for i in range(4):
        self.trackers[i] = cv2.TrackerKCF_create()
        r = model.rect_4patch(self.last_pos, i)
        self.trackers[i].init(img, r)
        self.lp[i] = (r[0] + r[2] / 2, r[1] + r[3] / 2)
        self.tot_dist[i] = 0.0
        self.bboxes[i] = r
    '''
    Note: another approach is to re-start the tracker only when the correlation score fall below some threshold
    i.e.: if bbox !=[] and self.confidence < 10.
    but this will reduce the algo. ability to track objects through longer periods of occlusions.
    '''

  def get_state(self):
    if self.last_pos:
      r = self.last_pos
      return [r[0], r[1], r[0] + r[2], r[1] + r[3]]
    else:
      return None
