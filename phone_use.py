"""
Mobile-phone use in class (Zayed) - MOBILE_PHONE_DETECTED.

No model of its own: YOLO26-L already scores COCO "cell phone" (67) on every frame of a camera
with mobile_phone_detection armed, and the main loop hands this module those boxes together with
the frame's person tracks.

A phone box is ATTACHED to the person whose box contains the phone's centre within the upper
PHONE_REGION_TOP_FRACTION of that box - hands at desk height and the face - choosing the
smallest such box (the nearest person when boxes overlap). A person is reported when a phone
stays attached for at least PHONE_MIN_SECONDS within the last PHONE_WINDOW_SECONDS, over at least
PHONE_MIN_HITS inferred frames. One event per person track per PHONE_COOLDOWN_SECONDS; the
evidence still is the person's box grown to include the phone.

HEURISTIC - pending validation on the real C101 cameras:
  * a phone lying on the desk inside a seated student's box also attaches;
  * at 640 px input a phone at the back of the room is only ~10-20 px, so recall falls with
    distance - PHONE_CONFIDENCE is therefore below the camera floor (as abandoned.py does for
    bags), and the persistence rule is what keeps single false boxes out.
"""
import os
import threading
from collections import deque

PHONE_CLASS_ID = 67
PHONE_FEATURE = "mobile_phone_detection"
EVENT_TYPE = "MOBILE_PHONE_DETECTED"

PHONE_CONFIDENCE = float(os.getenv("PHONE_CONFIDENCE", "0.30"))
PHONE_MIN_SECONDS = float(os.getenv("PHONE_MIN_SECONDS", "5"))
PHONE_WINDOW_SECONDS = float(os.getenv("PHONE_WINDOW_SECONDS", "15"))
PHONE_MIN_HITS = int(os.getenv("PHONE_MIN_HITS", "6"))
PHONE_COOLDOWN_SECONDS = float(os.getenv("PHONE_COOLDOWN_SECONDS", "300"))
PHONE_REGION_TOP_FRACTION = float(os.getenv("PHONE_REGION_TOP_FRACTION", "0.80"))
TRACK_FORGET_SECONDS = 60.0


def attach(phone_box, person_boxes):
    """Index of the person the phone belongs to, or None."""
    px = (phone_box[0] + phone_box[2]) / 2.0
    py = (phone_box[1] + phone_box[3]) / 2.0
    best, best_area = None, None
    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
        if not (x1 <= px <= x2):
            continue
        if not (y1 <= py <= y1 + PHONE_REGION_TOP_FRACTION * (y2 - y1)):
            continue
        area = max(1.0, (x2 - x1) * (y2 - y1))
        if best_area is None or area < best_area:
            best, best_area = i, area
    return best


def union(a, b):
    return [int(min(a[0], b[0])), int(min(a[1], b[1])), int(max(a[2], b[2])), int(max(a[3], b[3]))]


class PhoneFinding:
    def __init__(self, camera_id, track_id, bbox, phone_box, observed_at, hits, span, confidence):
        self.camera_id = camera_id
        self.track_id = track_id
        self.bbox = bbox
        self.phone_box = phone_box
        self.observed_at = observed_at
        self.hits = hits
        self.span = span
        self.confidence = confidence

    def scope(self):
        return f"phone:{self.track_id}"

    def metadata(self):
        return {"track_id": self.track_id, "phone_confidence": round(self.confidence, 3),
                "phone_bbox": [int(v) for v in self.phone_box], "hits": self.hits,
                "duration_seconds": round(self.span, 1), "window_seconds": PHONE_WINDOW_SECONDS,
                "method": "coco_cell_phone_attached_to_person",
                "semantic_note": "phone visible in the student's hand/face region - heuristic"}


class PhoneUseDetector:
    """Per-camera, per-track persistence of phone-to-person attachment. Main-thread only
    (update/forget), except set_enabled_cameras which the config refresh may call."""

    def __init__(self):
        self._enabled = frozenset()
        self._lock = threading.Lock()
        self._hits = {}          # (camera, track) -> deque[(t, confidence, phone_box)]
        self._last_seen = {}     # (camera, track) -> t
        self._last_event = {}    # (camera, track) -> t
        self.raised = 0
        self.attached = 0

    def set_enabled_cameras(self, camera_ids):
        with self._lock:
            self._enabled = frozenset(camera_ids or ())

    def enabled_for(self, camera_id):
        return camera_id in self._enabled

    def any_enabled(self):
        return bool(self._enabled)

    def update(self, camera_id, person_tracks, phone_detections, now):
        """person_tracks: TrackedObject list (person group). phone_detections: [(conf, box)].
        Returns [PhoneFinding] for persons that JUST met the rule."""
        if not self.enabled_for(camera_id):
            return []
        boxes = [t.bbox for t in person_tracks]
        for track in person_tracks:
            self._last_seen[(camera_id, track.track_id)] = now
        for confidence, phone_box in phone_detections:
            if confidence < PHONE_CONFIDENCE:
                continue
            index = attach(phone_box, boxes)
            if index is None:
                continue
            key = (camera_id, person_tracks[index].track_id)
            self._hits.setdefault(key, deque()).append((now, confidence, phone_box))
            self.attached += 1

        findings = []
        for track in person_tracks:
            key = (camera_id, track.track_id)
            hits = self._hits.get(key)
            if not hits:
                continue
            while hits and now - hits[0][0] > PHONE_WINDOW_SECONDS:
                hits.popleft()
            if len(hits) < PHONE_MIN_HITS:
                continue
            span = hits[-1][0] - hits[0][0]
            if span < PHONE_MIN_SECONDS:
                continue
            last = self._last_event.get(key)
            if last is not None and now - last < PHONE_COOLDOWN_SECONDS:
                continue
            self._last_event[key] = now
            best = max(hits, key=lambda h: h[1])
            findings.append(PhoneFinding(camera_id, track.track_id, union(track.bbox, best[2]), best[2],
                                         now, len(hits), span, best[1]))
            hits.clear()
            self.raised += 1
        return findings

    def forget_camera(self, camera_id):
        for store in (self._hits, self._last_seen, self._last_event):
            for key in [k for k in store if k[0] == camera_id]:
                del store[key]

    def prune(self, now):
        gone = [k for k, t in self._last_seen.items() if now - t > TRACK_FORGET_SECONDS]
        for key in gone:
            self._last_seen.pop(key, None)
            self._hits.pop(key, None)
        for key in [k for k, t in self._last_event.items() if now - t > PHONE_COOLDOWN_SECONDS]:
            del self._last_event[key]

    def stats(self):
        return {"cameras": sorted(self._enabled), "attached": self.attached, "raised": self.raised,
                "tracked": len(self._hits)}
