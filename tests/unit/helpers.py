import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class Track:
    def __init__(self, track_id, bbox, group="person", class_id=0):
        self.track_id, self.bbox, self.group, self.class_id = track_id, list(bbox), group, class_id


class Config:
    """Shaped like dashboard.CameraConfig for the fields the Zayed modules read."""

    def __init__(self, camera_id, classroom_id="C101", features=None, capacity=None, homography=None,
                 floor_plan=None, threshold_pct=100.0):
        self.camera_id = camera_id
        self.features = features or {}
        self.classroom = {"classroom_id": classroom_id, "capacity": capacity,
                          "overcrowding_threshold_pct": threshold_pct, "floor_plan": floor_plan or {}}
        self.placement = {"floor_plan_homography": homography or {}}
