import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(boxAArea + boxBArea - inter)

class Track:
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.disappeared = 0
        self.hits = 1          # how many frames it’s been matched
        self.confirmed = False

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.box
        return int((x1+x2)/2), int((y1+y2)/2)

class ImprovedTracker:
    def __init__(self, max_disappeared=10, min_hits=3, iou_thresh=0.3, dist_thresh=50):
        self.nextID = 0
        self.tracks = OrderedDict()
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh

    def register(self, box):
        self.tracks[self.nextID] = Track(self.nextID, box)
        self.nextID += 1

    def deregister(self, tid):
        del self.tracks[tid]

    def update(self, detections):
        # detections = [(x1,y1,x2,y2), ...]
        if len(detections) == 0:
            for t in list(self.tracks.values()):
                t.disappeared += 1
                if t.disappeared > self.max_disappeared:
                    self.deregister(t.track_id)
            return self.tracks

        # If no existing tracks, register all
        if len(self.tracks) == 0:
            for det in detections:
                self.register(det)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].box for tid in track_ids]

        # IoU matrix
        iou_mat = np.zeros((len(track_boxes), len(detections)), dtype=np.float32)
        for i, tb in enumerate(track_boxes):
            for j, det in enumerate(detections):
                iou_mat[i, j] = iou(tb, det)

        assigned_tracks = set()
        assigned_dets = set()

        # First pass: IoU > threshold
        for i in range(len(track_boxes)):
            j = np.argmax(iou_mat[i])
            if iou_mat[i, j] >= self.iou_thresh and j not in assigned_dets:
                t = self.tracks[track_ids[i]]
                t.box = detections[j]
                t.disappeared = 0
                t.hits += 1
                if t.hits >= self.min_hits:
                    t.confirmed = True
                assigned_tracks.add(i)
                assigned_dets.add(j)

        # Second pass: centroid distance for unassigned
        for i in range(len(track_boxes)):
            if i in assigned_tracks:
                continue
            t = self.tracks[track_ids[i]]
            t_cent = t.centroid
            best_j = -1
            best_d = self.dist_thresh
            for j, det in enumerate(detections):
                if j in assigned_dets:
                    continue
                cX = (det[0]+det[2])//2
                cY = (det[1]+det[3])//2
                d = dist.euclidean(t_cent, (cX,cY))
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j != -1:
                t.box = detections[best_j]
                t.disappeared = 0
                t.hits += 1
                if t.hits >= self.min_hits:
                    t.confirmed = True
                assigned_tracks.add(i)
                assigned_dets.add(best_j)
            else:
                t.disappeared += 1
                if t.disappeared > self.max_disappeared:
                    self.deregister(t.track_id)

        # Register any remaining detections
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self.register(det)

        return self.tracks

