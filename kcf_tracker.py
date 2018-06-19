"""
@author: Mahmoud I.Zidan
"""

import cv2
import numpy as np

'''Appearance Model'''
class KCFTracker:

  count = 0
  def __init__(self,bbox,img):
    self.tracker = cv2.TrackerKCF_create()
    self.last_pos = (int(bbox[0]),int(bbox[1]),int(bbox[2] - bbox[0]),int(bbox[3] - bbox[1]))
    self.tracker.init(img, self.last_pos)
    self.tracker.update(img)
    self.confidence = 0. # measures how confident the tracker is! (a.k.a. correlation score)

    self.time_since_update = 0
    self.id = KCFTracker.count
    KCFTracker.count += 1
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.lost = False

  def predict(self,img):
    ok, bbox = self.tracker.update(img)
    if ok:
        self.lost = False
        self.last_pos = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
    else:
        self.lost = True
        #self.last_pos = None

    self.age += 1
    if (self.time_since_update > 0):
      self.hit_streak = 0
    self.time_since_update += 1

    return self.get_state()

  def update(self,bbox,img):
    if bbox != []:
     self.time_since_update = 0
     self.hits += 1
     self.hit_streak += 1

    '''re-start the tracker with detected positions (it detector was active)'''
    if bbox != []:
      self.tracker = cv2.TrackerKCF_create()
      self.last_pos = (int(bbox[0]),int(bbox[1]),int(bbox[2] - bbox[0]),int(bbox[3] - bbox[1]))
      self.tracker.init(img, self.last_pos)
      self.tracker.update(img)
    '''
    Note: another approach is to re-start the tracker only when the correlation score fall below some threshold
    i.e.: if bbox !=[] and self.confidence < 10.
    but this will reduce the algo. ability to track objects through longer periods of occlusions.
    '''

  def get_state(self):
    if self.last_pos:
      return [self.last_pos[0], self.last_pos[1], self.last_pos[0] + self.last_pos[2], self.last_pos[1] + self.last_pos[3]]
    else:
      return None
