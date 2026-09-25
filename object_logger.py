"""
Object detection logging - one event per physical object sighting.

    logger = ObjectLogger()
    logger.set_enabled_cameras({"CAM-R25": class_ids_for(["backpack", "knife"])})
    for tracked, metadata, crop_box in logger.observe(camera_id, object_tracks,
                                                       width, height, now):
        event_pipeline.handle_object_event(...)

WHAT IT LOGS
------------
YOLO26-L already scores all 80 COCO classes on every frame; until this module
the pipeline kept eight of them and discarded the rest. An armed camera now
also keeps the object classes its dashboard configuration names, tracks them in
the tracker's object group, and this module turns each one into ONE
object_detected event with a picture.

Persons and the five vehicle classes are deliberately not loggable here: they
have their own analytics and event types (person_detected, vehicle_detected).

ONE EVENT PER OBJECT, NOT PER FRAME
-----------------------------------
person_detected is presence sampling and produced 64,657 rows in a day. An
object log must not repeat that, and a raw "new track" rule is not enough,
for two reasons measured on this estate:

  * A detection must survive CONFIRM_FRAMES matched frames (JeztSort's own
    min_hits is only 3) before it is logged, so a one-off false box does not
    become a permanent row with a picture.
  * Tracks fragment. A bag behind a passer-by, or a chair that flickers under
    the confidence floor, comes back under a NEW track id. A new track of the
    same class that overlaps a sighting last seen within REDETECT_SECONDS is
    the same object: it is attached to that sighting and nothing is logged.
    A static object that is re-detected all day is therefore logged once,
    until it has been gone for REDETECT_SECONDS.

A sighting that a live track updated in the current frame is never matched by
another track in that frame: two objects visible at the same moment are two
objects, however close.

THE PICTURE
-----------
Small objects are where the evidence matters most and where a padded bbox crop
fails: a phone is 30x60 px, under the evidence minimum once padded. The crop
box returned here is the object's box grown to CONTEXT_SCALE times its size and
at least CONTEXT_MIN_PX on each side, so the still shows the object and what it
is on or near.

Env overrides:
  OBJECT_CONFIRM_FRAMES     matched frames before an object is logged  (5)
  OBJECT_REDETECT_SECONDS   gap after which the same spot logs again (120)
  OBJECT_MATCH_IOU          overlap that makes a new track the same object (0.3)
"""
import os
import threading

#: The dashboard feature flag that arms object logging for a camera.
OBJECT_DETECTION_FEATURE = "object_detection"

#: The dashboard event type.
EVENT_TYPE = "object_detected"

#: COCO class names in model index order, exactly as yolo26l.pt names them.
#: multicam_inf.py checks this against the loaded model at startup.
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

CLASS_ID = {name: index for index, name in enumerate(COCO_NAMES)}

#: person, bicycle, car, motorcycle, bus, truck - owned by other analytics.
NOT_LOGGABLE = frozenset((0, 1, 2, 3, 5, 7))

LOGGABLE_CLASS_IDS = frozenset(
    index for index in range(len(COCO_NAMES)) if index not in NOT_LOGGABLE)

#: Used when a camera arms object detection without choosing classes. Things a
#: person carries, leaves or brings in, plus animals (an RFP false-alarm
#: source). Furniture and fixtures are left out: they are part of the room.
DEFAULT_OBJECT_CLASSES = (
    "backpack", "handbag", "suitcase", "umbrella", "bottle", "cup",
    "cell phone", "laptop", "knife", "scissors", "baseball bat",
    "bird", "cat", "dog",
)

CONFIRM_FRAMES = int(os.getenv("OBJECT_CONFIRM_FRAMES", "5"))
REDETECT_SECONDS = float(os.getenv("OBJECT_REDETECT_SECONDS", "120"))
MATCH_IOU = float(os.getenv("OBJECT_MATCH_IOU", "0.3"))

#: A moving object that fragments has moved by the time it is re-acquired, so
#: overlap alone misses it. Within this short gap, a centre within
#: MOVE_MATCH_DIAGONALS of the old box's diagonal also counts as the same one.
MOVE_MATCH_SECONDS = 3.0
MOVE_MATCH_DIAGONALS = 1.5

CONTEXT_SCALE = 2.5
CONTEXT_MIN_PX = 160

#: Per-track bookkeeping for a track that stopped arriving is dropped after this.
TRACK_STATE_MAX_AGE_SECONDS = 30.0


def class_ids_for(names):
    """Class names -> loggable COCO ids.

    Names that are unknown or not loggable are dropped FIRST, and a list with
    nothing usable left means the defaults - the same as no list at all - so a
    camera armed for objects can never end up silently logging nothing.
    """
    chosen = frozenset(CLASS_ID[name] for name in (names or ())
                       if name in CLASS_ID) & LOGGABLE_CLASS_IDS
    return chosen or frozenset(CLASS_ID[name] for name in DEFAULT_OBJECT_CLASSES)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _centre_distance(a, b):
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _diagonal(box):
    return ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5


def context_box(bbox, frame_width, frame_height,
                scale=CONTEXT_SCALE, min_px=CONTEXT_MIN_PX):
    """The object's box grown for context, clamped to the frame."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max((x2 - x1) * scale, min_px) / 2.0
    half_h = max((y2 - y1) * scale, min_px) / 2.0
    return [
        max(0, int(cx - half_w)),
        max(0, int(cy - half_h)),
        min(int(frame_width), int(cx + half_w)),
        min(int(frame_height), int(cy + half_h)),
    ]


class ObjectLogger:
    """Per-camera object sightings, fed with each frame's object tracks."""

    def __init__(self, log=print):
        self._log = log
        self._lock = threading.Lock()

        # camera_id -> frozenset of armed class ids. Replaced wholesale.
        self._classes = {}

        # (camera_id, track_id) -> {"frames", "last_seen", "sighting"}
        self._tracks = {}

        # camera_id -> {sighting_id: {"class_id", "bbox", "last_seen", "updated_at"}}
        self._sightings = {}
        self._next_sighting = 0

        self.logged = 0
        self.reattached = 0         # a new track recognised as an existing sighting
        self.by_class = {}
        self.errors = 0
        self._error_logged = False

    # --------------------------------------------------------------- config

    def set_enabled_cameras(self, camera_classes):
        """Arm exactly these cameras: {camera_id: frozenset(class ids)}."""
        armed = {camera_id: frozenset(ids) & LOGGABLE_CLASS_IDS
                 for camera_id, ids in camera_classes.items()}
        with self._lock:
            self._classes = armed
            for camera_id in [c for c in self._sightings if c not in armed]:
                del self._sightings[camera_id]
            for key in [k for k in self._tracks if k[0] not in armed]:
                del self._tracks[key]

    def enabled_cameras(self):
        return sorted(self._classes)

    def class_ids(self, camera_id):
        return self._classes.get(camera_id, frozenset())

    # -------------------------------------------------------------- observe

    def observe(self, camera_id, object_tracks, frame_width, frame_height, now):
        """One frame. Returns [(tracked, metadata, crop_box)] to log. Never raises."""
        armed = self._classes.get(camera_id)
        if not armed:
            return []

        try:
            with self._lock:
                return self._observe(camera_id, armed, object_tracks,
                                     frame_width, frame_height, float(now))
        except Exception as exc:                                 # noqa: BLE001
            self.errors += 1
            if not self._error_logged:
                self._error_logged = True
                self._log(f"[OBJECT_LOG] observe error (logged once): "
                          f"{type(exc).__name__}: {exc}")
            return []

    def _observe(self, camera_id, armed, object_tracks, frame_width, frame_height, now):
        sightings = self._sightings.setdefault(camera_id, {})
        tracks = [t for t in object_tracks if t.class_id in armed]

        # Pass 1: tracks already attached to a sighting keep it current, so
        # pass 2 can tell which sightings are visibly occupied this frame.
        pending = []
        for tracked in tracks:
            key = (camera_id, tracked.track_id)
            state = self._tracks.get(key)
            if state is None:
                state = self._tracks[key] = {"frames": 0, "last_seen": now, "sighting": None}
            state["frames"] += 1
            state["last_seen"] = now

            sighting = sightings.get(state["sighting"])
            if sighting is not None:
                sighting["bbox"] = list(tracked.bbox)
                sighting["last_seen"] = now
                sighting["updated_at"] = now
            elif state["frames"] >= CONFIRM_FRAMES:
                pending.append((tracked, state))

        # Pass 2: confirmed tracks with no sighting yet - an existing object
        # under a new track id, or a new object.
        fired = []
        for tracked, state in pending:
            match = self._match(sightings, tracked, now)
            if match is not None:
                state["sighting"] = match
                sightings[match].update(bbox=list(tracked.bbox), last_seen=now, updated_at=now)
                self.reattached += 1
                continue

            self._next_sighting += 1
            sighting_id = self._next_sighting
            sightings[sighting_id] = {"class_id": tracked.class_id, "bbox": list(tracked.bbox),
                                      "last_seen": now, "updated_at": now}
            state["sighting"] = sighting_id

            self.logged += 1
            self.by_class[tracked.class_name] = self.by_class.get(tracked.class_name, 0) + 1
            fired.append((
                tracked,
                {"class_id": int(tracked.class_id), "confirm_frames": state["frames"]},
                context_box(tracked.bbox, frame_width, frame_height),
            ))

        self._prune(camera_id, sightings, now)
        return fired

    @staticmethod
    def _match(sightings, tracked, now):
        best, best_score = None, 0.0
        for sighting_id, sighting in sightings.items():
            if sighting["class_id"] != tracked.class_id:
                continue
            if sighting["updated_at"] == now:          # visibly another object
                continue
            gap = now - sighting["last_seen"]
            if gap > REDETECT_SECONDS:
                continue
            score = _iou(sighting["bbox"], tracked.bbox)
            if score < MATCH_IOU:
                near = _centre_distance(sighting["bbox"], tracked.bbox)
                if not (gap <= MOVE_MATCH_SECONDS
                        and near <= MOVE_MATCH_DIAGONALS * _diagonal(sighting["bbox"])):
                    continue
                score = MATCH_IOU                      # weakest acceptable match
            if score > best_score:
                best, best_score = sighting_id, score
        return best

    def _prune(self, camera_id, sightings, now):
        for sighting_id in [s for s, v in sightings.items()
                            if now - v["last_seen"] > REDETECT_SECONDS]:
            del sightings[sighting_id]
        for key in [k for k, v in self._tracks.items()
                    if k[0] == camera_id and now - v["last_seen"] > TRACK_STATE_MAX_AGE_SECONDS]:
            del self._tracks[key]

    # ---------------------------------------------------------------- stats

    def status_line(self):
        with self._lock:
            top = sorted(self.by_class.items(), key=lambda kv: -kv[1])[:5]
            live = sum(len(s) for s in self._sightings.values())
        return (f"ObjectLog: cameras={len(self._classes)} logged={self.logged} "
                f"reattached={self.reattached} live_sightings={live} "
                f"errors={self.errors} top={dict(top)}")
