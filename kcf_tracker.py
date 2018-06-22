import cv2
import numpy as np
import model

class KCFTracker:
    count = 0

    def __init__(self, bbox, img, new_id, use_4patch, stop_on_4patch_break):
        self.num_patches = 4 if use_4patch else 0
        self.stop_on_4patch_break = stop_on_4patch_break

        self.last_pos = (int(bbox[0]),int(bbox[1]),int(bbox[2] - bbox[0]),int(bbox[3] - bbox[1]))
        if self.num_patches == 0:
            self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(img, self.last_pos)
            self.tracker.update(img)
        else:
            self.trackers = [cv2.TrackerKCF_create(), cv2.TrackerKCF_create(), cv2.TrackerKCF_create(), cv2.TrackerKCF_create()]
            self.lp = [0, 0, 0, 0]
            self.tot_dist = [0.0, 0.0, 0.0, 0.0]
            self.bboxes = [0, 0, 0, 0]
            for i in range(self.num_patches):
                r = model.rect_4patch(self.last_pos, i)
                self.trackers[i].init(img, r)
                self.trackers[i].update(img)
                self.lp[i] = (r[0] + r[2] / 2, r[1] + r[3] / 2)
                self.bboxes[i] = r

        self.time_since_update = 0
        if new_id:
            self.id = new_id
        else:
            self.id = KCFTracker.count
            KCFTracker.count += 1
        self.points = []
        self.mixed_ids = []

    def predict(self,img):
        if self.num_patches == 0:
            ok, bbox = self.tracker.update(img)
            if ok:
                self.last_pos = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
                self.points.append((int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)))
                if len(self.points) > 100:
                    del self.points[0]
        else:
            frame = img.copy()
            for i in range(self.num_patches):
                ok, bbox = self.trackers[i].update(img)
                if ok:
                    self.bboxes[i] = bbox
                    ps = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
                    diff = model.pt_dist(ps, self.lp[i])
                    self.lp[i] = ps
                    self.tot_dist[i] += diff
                    cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[0] + bbox[2]),int(bbox[1] + bbox[3])), (255,0,0), 3)

            idx = list(np.argsort(self.tot_dist))

            break_detected = False

            for i in range(self.num_patches):
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
                    if i != 0:
                        for k in range(self.num_patches):
                            self.trackers[k] = cv2.TrackerKCF_create()
                            r = model.rect_4patch(rct_good, k)
                            self.trackers[k].init(img, r)
                            self.trackers[k].update(img)
                            self.lp[k] = (r[0] + r[2] / 2, r[1] + r[3] / 2)
                            self.tot_dist[k] = 0.0
                            self.bboxes[k] = r
                        cv2.rectangle(frame, (int(rct_good[0]),int(rct_good[1])), (int(rct_good[0] + rct_good[2]),int(rct_good[1] + rct_good[3])), (0,0,255), 2)
                        break_detected = True
                    break
                del idx[0]

            cv2.imshow('frame_4patch', frame)
            if break_detected and self.stop_on_4patch_break:
                cv2.waitKey()

        self.time_since_update += 1

        return self.get_state()

    def update(self,bbox,img):
        if bbox != []:
            self.time_since_update = 0

            self.last_pos = (int(bbox[0]),int(bbox[1]),int(bbox[2] - bbox[0]),int(bbox[3] - bbox[1]))
            if self.num_patches == 0:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(img, self.last_pos)
                self.tracker.update(img)
            else:
                for i in range(self.num_patches):
                    self.trackers[i] = cv2.TrackerKCF_create()
                    r = model.rect_4patch(self.last_pos, i)
                    self.trackers[i].init(img, r)
                    self.trackers[i].update(img)
                    self.lp[i] = (r[0] + r[2] / 2, r[1] + r[3] / 2)
                    self.tot_dist[i] = 0.0
                    self.bboxes[i] = r

    def get_state(self):
        r = self.last_pos
        return [r[0], r[1], r[0] + r[2], r[1] + r[3]]
