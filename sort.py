"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function
import numpy as np
from kcf_tracker import KCFTracker
from data_association import associate_detections_to_trackers
import model

class Sort:
    def __init__(self, use_4patch, stop_on_4patch_break, max_age, mix_threshold, inflate_ratio, mix_life_threshold):
        """
        Sets key parameters for SORT
        """
        self.use_4patch = use_4patch
        self.stop_on_4patch_break = stop_on_4patch_break
        self.max_age = max_age
        self.mix_threshold = mix_threshold
        self.inflate_ratio = inflate_ratio
        self.mix_life_threshold = mix_life_threshold
        self.trackers = []

    def update(self,dets,img=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict(img)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        if dets != []:
            dets_inflated = np.copy(dets)
            for i in range(dets_inflated.shape[0]):
                r1 = dets_inflated[i, :]
                r1 = (r1[0], r1[1], r1[2] - r1[0], r1[3] - r1[1])
                r1 = model.rect_inflate2(r1, r1[2] * self.inflate_ratio, r1[3] * self.inflate_ratio)
                dets_inflated[i][0] = r1[0]
                dets_inflated[i][1] = r1[1]
                dets_inflated[i][2] = r1[2] + r1[0]
                dets_inflated[i][3] = r1[3] + r1[1]
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_inflated,trks)

            #update matched trackers with assigned detections
            for t,trk in enumerate(self.trackers):
                if (t not in unmatched_trks):
                    d = matched[np.where(matched[:,1]==t)[0],0]
                    trk.update(dets[d,:][0],img)

            #create and initialize new trackers for unmatched detections
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
                        if area_factor > self.mix_threshold:
                            tr.time_since_update = 0
                            skip = True
                            break
                if skip:
                    continue
                new_id = None
                for tr in self.trackers:
                    r2 = dets[i,:]
                    r2 = (r2[0], r2[1], r2[2] - r2[0], r2[3] - r2[1])
                    if tr.mixed_ids:
                        r3 = tr.get_state()
                        r3 = (r3[0], r3[1], r3[2] - r3[0], r3[3] - r3[1])
                        r3 = model.rect_inflate2(r3, max(r3[2], r3[3]), max(r3[2], r3[3]))
                        isect = model.rect_intersection(r2, r3)
                        if isect:
                            new_id = tr.mixed_ids[0]
                            del tr.mixed_ids[0]
                            break
                trk = KCFTracker(dets[i,:], img, new_id, self.use_4patch, self.stop_on_4patch_break)
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([],img)
            i -= 1
            #remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        i = len(self.trackers)
        for trk1 in reversed(self.trackers):
            i -= 1
            if trk1.life < self.mix_life_threshold:
                continue
            r1 = trk1.get_state()
            r1 = (r1[0], r1[1], r1[2] - r1[0], r1[3] - r1[1])
            for trk2 in self.trackers:
                if trk1 == trk2:
                    continue
                if trk2.life < self.mix_life_threshold:
                    continue
                r2 = trk2.get_state()
                r2 = (r2[0], r2[1], r2[2] - r2[0], r2[3] - r2[1])
                isect = model.rect_intersection(r1, r2)
                if isect:
                    area_factor = (isect[2] * isect[3]) / min(r1[2] * r1[3], r2[2] * r2[3])
                    if area_factor > self.mix_threshold:
                        trk2.mixed_ids += trk1.mixed_ids
                        trk2.mixed_ids.append(trk1.id)
                        self.trackers.pop(i)
                        break
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            ret.append(list(d) + [trk.id, trk.points, trk.mixed_ids])

        return ret
