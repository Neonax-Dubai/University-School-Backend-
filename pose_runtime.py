"""
Shared pose engine for the asynchronous analytics adapters.

    runtime = PoseRuntime()          # constructed at import, loads nothing
    runtime.acquire()                # first armed camera - builds the engine
    people = runtime.infer(frame, tracks)

WHY THIS EXISTS
---------------
fight_adapter.py and fence_adapter.py both need COCO 17-keypoint pose, and
nothing else. The two prototypes happened to use different engines
(yolo26m-pose and yolo26l-pose respectively), but test_fence_det.py's own
docstring says its choice was "the SAME one test_behavior.py uses, as
requested" - a convenience, not a requirement. Loading two engines costs a
second ~800 MB of GPU memory and a second load, to compute the same 17 points.
This loads ONE engine and both adapters share it.

IDENTITY COMES FROM PRODUCTION, NOT FROM HERE
---------------------------------------------
This does NOT run a tracker. It is handed the tracks multicam_inf.py already
produced and matches this frame's keypoints back onto those boxes by IOU - the
same technique fight_fall_prototype.build_people() uses, minus its TrackManager
call. So there is exactly one person detector and one tracker in the process,
both of them the existing production ones.

The engine is built lazily on first acquire() and never rebuilt: a failure is
latched, so a missing or broken model does not get retried on every config
refresh.
"""

import os
import threading
import time

POSE_MODEL = os.getenv("POSE_ADAPTER_MODEL", "models/yolo26m-pose.engine")
POSE_MODEL_FALLBACK = os.getenv("POSE_ADAPTER_MODEL_PT", "models/yolo26m-pose.pt")
POSE_IMGSZ = int(os.getenv("POSE_ADAPTER_IMGSZ", "640"))
POSE_CONF = float(os.getenv("POSE_ADAPTER_CONF", "0.35"))
POSE_MATCH_IOU = float(os.getenv("POSE_ADAPTER_MATCH_IOU", "0.3"))

# Result distribution. One frame -> one inference -> N consumers.
# Keyed on (camera_id, frame_id): cctv.py gives every CameraStream its own
# monotonic frame_id (cctv.py:224), so the pair identifies a frame exactly,
# with no reliance on wall-clock timestamps that two cameras can share.
POSE_CACHE_SIZE = int(os.getenv("POSE_CACHE_SIZE", "64"))
POSE_CACHE_TTL_SECONDS = float(os.getenv("POSE_CACHE_TTL_SECONDS", "2.0"))


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class PoseRuntime:
    """One pose engine, shared by every adapter that needs keypoints."""

    def __init__(self, model_factory=None):
        self._lock = threading.Lock()
        self._model = None
        self._failed = False
        self._users = 0
        self._factory = model_factory or _load_pose_model
        self.inferences = 0
        self.errors = 0
        # (camera_id, frame_id) -> (monotonic_inserted_at, people, threading.Event)
        # An Event is stored BEFORE the inference starts, so a second consumer
        # asking for the same frame waits for the first one's result instead of
        # computing it again. This is what turns N consumers into 1 inference.
        self._cache = {}
        self._cache_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_waits = 0
        self.cache_evictions = 0

    # ---------------------------------------------------------- lifecycle
    def acquire(self):
        """Called when an adapter arms its first camera. Builds on first use."""
        with self._lock:
            self._users += 1
            if self._model is not None or self._failed:
                return self._model is not None
            try:
                self._model = self._factory()
            except Exception as exc:  # noqa: BLE001
                # Latched: a broken model must not be retried every 30 s.
                self._failed = True
                print(f"[POSE] model unavailable, pose analytics disabled: "
                      f"{type(exc).__name__}: {exc}")
                return False
            print(f"[POSE] shared pose engine ready: {POSE_MODEL} imgsz={POSE_IMGSZ}")
            return True

    def release(self):
        with self._lock:
            self._users = max(0, self._users - 1)

    @property
    def available(self):
        return self._model is not None

    # ---------------------------------------------------------- inference
    def infer(self, frame, tracks, camera_id=None, frame_id=None):
        """Run pose on one frame and attach keypoints to production tracks.

        tracks : iterable of (track_id, bbox) already produced by the pipeline.
        Returns [(track_id, kpts(17,2), confs(17,), bbox)] - the exact tuple
        shape fight_fall_prototype/fence_detection's own functions consume.

        When camera_id AND frame_id are given, the result for that exact frame
        is computed ONCE and shared with every other consumer asking for it -
        so arming Fight and Fence on one camera costs one inference, not two.
        Without them the old behaviour is unchanged: infer every time.
        """
        model = self._model
        if model is None or not tracks:
            return []

        key = None if (camera_id is None or frame_id is None) else (camera_id, frame_id)
        if key is not None:
            now = time.monotonic()
            with self._cache_lock:
                self._evict(now)
                entry = self._cache.get(key)
                if entry is not None:
                    self.cache_hits += 1
                    ready = entry[2]
                else:
                    self.cache_misses += 1
                    ready = threading.Event()
                    self._cache[key] = [now, None, ready]
                    entry = None
            if entry is not None:
                # Someone else owns this frame. Wait briefly for their result
                # rather than running the same inference again. The timeout is
                # a safety valve, not an expectation.
                if not entry[2].is_set():
                    self.cache_waits += 1
                    entry[2].wait(timeout=POSE_CACHE_TTL_SECONDS)
                with self._cache_lock:
                    current = self._cache.get(key)
                return list(current[1]) if current and current[1] is not None else []

        try:
            import numpy as np
            res = model.predict(frame, conf=POSE_CONF, imgsz=POSE_IMGSZ,
                                verbose=False, classes=[0])[0]
            self.inferences += 1
            if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
                return []
            boxes = res.boxes.xyxy.cpu().numpy()
            kxy = res.keypoints.xy.cpu().numpy()
            kcf = (res.keypoints.conf.cpu().numpy()
                   if res.keypoints.conf is not None
                   else np.ones(kxy.shape[:2]))

            people = []
            for track_id, bbox in tracks:
                best_iou, best_i = 0.0, None
                for i, b in enumerate(boxes):
                    iou = _box_iou(bbox, b)
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                # An unmatched track is skipped rather than guessed at: giving a
                # person somebody else's skeleton is worse than no result.
                if best_i is None or best_iou < POSE_MATCH_IOU:
                    continue
                people.append((track_id, kxy[best_i], kcf[best_i],
                               np.array(bbox, dtype=float)))
            self._publish(key, people)
            return people
        except Exception as exc:  # noqa: BLE001 - never reaches the worker loop
            self.errors += 1
            print(f"[POSE] inference error (contained): {type(exc).__name__}: {exc}")
            # Release anyone waiting on this frame rather than making them
            # sit out the full timeout for a result that will never arrive.
            self._publish(key, [])
            return []

    def _publish(self, key, people):
        """Hand this frame's result to every consumer waiting on it."""
        if key is None:
            return
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return
            entry[1] = list(people)
            event = entry[2]
        event.set()

    def _evict(self, now):
        """Bound the cache by age and size. Caller holds the lock.

        Age first: a CameraStream that reconnects restarts its frame_id at 0
        (cctv.py:65), so an old entry for the same (camera, frame_id) must not
        be reusable. A TTL far shorter than any reconnect makes that safe.
        """
        stale = [k for k, v in self._cache.items()
                 if (now - v[0]) > POSE_CACHE_TTL_SECONDS]
        for k in stale:
            self._cache[k][2].set()      # never leave a waiter hanging
            del self._cache[k]
            self.cache_evictions += 1
        if len(self._cache) > POSE_CACHE_SIZE:
            for k in sorted(self._cache, key=lambda k: self._cache[k][0])[
                    :len(self._cache) - POSE_CACHE_SIZE]:
                self._cache[k][2].set()
                del self._cache[k]
                self.cache_evictions += 1

    def stats(self):
        return {"pose_loaded": self._model is not None,
                "pose_failed": self._failed,
                "pose_users": self._users,
                "pose_inferences": self.inferences,
                "pose_errors": self.errors,
                "pose_cache_hits": self.cache_hits,
                "pose_cache_misses": self.cache_misses,
                "pose_cache_waits": self.cache_waits,
                "pose_cache_evictions": self.cache_evictions,
                "pose_cache_size": len(self._cache)}


def _load_pose_model():
    from ultralytics import YOLO
    base = os.path.dirname(os.path.abspath(__file__))
    for candidate in (POSE_MODEL, POSE_MODEL_FALLBACK):
        path = candidate if os.path.isabs(candidate) else os.path.join(base, candidate)
        if os.path.exists(path):
            return YOLO(path, task="pose")
    raise FileNotFoundError(
        f"no pose model found: tried {POSE_MODEL} and {POSE_MODEL_FALLBACK}")


# One engine for the whole process.
shared_pose = PoseRuntime()
