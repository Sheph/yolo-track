"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function
import numpy as np
from kalman_tracker import KalmanBoxTracker
from correlation_tracker import CorrelationTracker
from kcf_tracker import KCFTracker
from data_association import associate_detections_to_trackers
import model


class Sort:

  def __init__(self,max_age=25*2,min_hits=3, use_dlib = False):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

    self.use_dlib = use_dlib

  def update(self,dets,img=None):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict(img) #for kal!
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    if dets != []:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

      #update matched trackers with assigned detections
      for t,trk in enumerate(self.trackers):
        if(t not in unmatched_trks):
          d = matched[np.where(matched[:,1]==t)[0],0]
          trk.update(dets[d,:][0],img) ## for dlib re-intialize the trackers ?!

      #create and initialise new trackers for unmatched detections
      for i in unmatched_dets:
        skip = False
        for tr in self.trackers:
            r1 = tr.get_state()
            r2 = dets[i,:]
            r1 = (r1[0], r1[1], r1[2] - r1[0], r1[3] - r1[1])
            r2 = (r2[0], r2[1], r2[2] - r2[0], r2[3] - r2[1])
            isect = model.rect_intersection(r1, r2)
            if isect:
                area_factor = (isect[2] * isect[3]) / min(r1[2] * r1[3], r2[2] * r2[3])
                if area_factor > 0.5:
                    tr.time_since_update = 0
                    skip = True
        if skip:
            continue
        if not self.use_dlib:
          trk = KalmanBoxTracker(dets[i,:])
        else:
          #trk = CorrelationTracker(dets[i,:],img)
          trk = KCFTracker(dets[i,:],img)
          #trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        if dets == []:
          trk.update([],img)
        d = trk.get_state()
        #if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
        ret.append(list(d) + [trk.id+1, trk.points]) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if trk.lost:
            ma = self.max_age
        else:
            ma = self.max_age * 4
        if(trk.time_since_update > ma):
          self.trackers.pop(i)
    if(len(ret)>0):
      return ret
    return []
