"""
Asynchronous pose Fall adapter - owns the pose model, never blocks a camera.

Cost class: FULLY ASYNCHRONOUS, the same shape as fight_adapter and
fence_adapter. `submit()` does a cheap geometric test and one `put_nowait`;
every millisecond of model work happens on a private worker thread. A camera
thread can never wait on the GPU, and a full queue drops the frame rather
than applying back-pressure.

    adapter.submit(camera_id, frame, tracks, timestamp, frame_id)   # camera thread
    for event in adapter.flush():                                   # main thread
        ...

WHAT STARTS A POSE EVALUATION (section 8)
-----------------------------------------
The bounding box is now a CANDIDATE TRIGGER ONLY. It selects which tracks are
worth spending pose on; it no longer decides anything.

A person track becomes a pose candidate when

    bbox aspect ratio (w/h) >= CANDIDATE_AR      (default 0.75)

i.e. "not clearly upright" - deliberately far more permissive than the old
fall_policy.GROUND_AR of 1.05 that was making the decision. A track that
becomes a candidate stays hot for CANDIDATE_HOLD_SECONDS afterwards, so a
momentary dip in aspect ratio cannot interrupt a confirmation window.

Measured against the 25 offline videos, this gate retains 248 of the 252
frames whose torso angle already exceeds the production threshold. The four
it drops all sit in one video, FALL_EVT-F39FB316 - a confirmed FALSE
POSITIVE. The known true positive's lowest aspect ratio at a >=75 deg frame
is 1.979, far above the gate. So the gate costs nothing that the decision
would have used, and it is the reason a pose model can be afforded here at
all.

MODEL
-----
YOLO26L-pose at 960, conf 0.25 - the exact configuration the offline
experiment validated, and the same weights file the Behaviour analytic
already runs. This adapter loads its own instance (~803 MB) rather than
sharing, because sharing would mean restructuring behaviour_adapter.py,
which is a protected file. Sharing is worth revisiting separately.

It deliberately does NOT use pose_runtime.shared_pose: that runtime is
yolo26m-pose at 640 with conf 0.35, and the Fall thresholds below were
validated at L/960/0.25. Reusing it would silently invalidate that.
"""

import os
import queue
import threading
import time
from collections import defaultdict

import fall_pose_policy as policy
import pose_l960_runtime
from fall_pose_policy import (COOLDOWN, EVENT_TYPE, FALL_CAMERA_FEATURE,  # noqa: F401
                              NORMAL, POSE_CANDIDATE, POSE_CONFIRMED,
                              POSE_CONFIRMING, POSE_UNCERTAIN, FallPosePolicy)

PERSON_GROUP = "person"


def _bool_env(name, default):
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _float_env(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


FALL_POSE_ENABLED = _bool_env("FALL_POSE_ENABLED", True)
POSE_MODEL = os.getenv("FALL_POSE_MODEL",
                       "distance_measuring/models/yolo26l-pose.engine")
POSE_MODEL_FALLBACK = os.getenv("FALL_POSE_MODEL_PT",
                                "distance_measuring/models/yolo26l-pose.pt")
POSE_IMGSZ = _int_env("FALL_POSE_IMGSZ", 960)
POSE_CONF = _float_env("FALL_POSE_CONF", 0.25)
POSE_MATCH_IOU = _float_env("FALL_POSE_MATCH_IOU", 0.30)

CANDIDATE_AR = _float_env("FALL_POSE_CANDIDATE_AR", 0.75)
CANDIDATE_HOLD_SECONDS = _float_env("FALL_POSE_CANDIDATE_HOLD_SECONDS", 2.0)

QUEUE_SIZE = _int_env("FALL_POSE_QUEUE_SIZE", 8)
#: Minimum wall-clock gap between two submissions for one camera. 0.0 means
#: "every candidate frame", which is what the offline experiment did.
INTERVAL_SECONDS = _float_env("FALL_POSE_INTERVAL_SECONDS", 0.0)

#: Shadow mode (section 24): evaluate and log everything, emit no event.
SHADOW_MODE = _bool_env("FALL_POSE_SHADOW_MODE", False)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter / ua) if ua > 0 else 0.0


def is_candidate(bbox):
    """The section-8 gate. Pure geometry, safe to call on a camera thread."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    h = y2 - y1
    if h <= 0:
        return False
    return ((x2 - x1) / h) >= CANDIDATE_AR


def unpack_pose(res):
    """ultralytics Result -> (boxes, keypoints_xy, keypoint_conf) or Nones."""
    if res is None or res.boxes is None or not len(res.boxes) \
            or res.keypoints is None:
        return None, None, None
    boxes = res.boxes.xyxy.cpu().numpy()
    kxy = res.keypoints.xy.cpu().numpy()
    if res.keypoints.conf is not None:
        kcf = res.keypoints.conf.cpu().numpy()
    else:
        import numpy as np
        kcf = np.ones(kxy.shape[:2])
    return boxes, kxy, kcf


class _Stats:
    def __init__(self):
        self.submitted = 0
        self.processed = 0
        self.candidates = 0
        self.skipped_not_candidate = 0
        self.skipped_interval = 0
        self.skipped_not_armed = 0
        self.dropped_queue_full = 0
        self.worker_errors = 0
        self.model_errors = 0
        self.pose_persons = 0
        self.unassociated_tracks = 0
        self.contested_poses = 0
        self.events = 0
        self.shadow_events = 0
        self.infer_ms_total = 0.0
        self.infer_n = 0

    def as_dict(self):
        d = {f"fall_pose_{k}": v for k, v in vars(self).items()
             if not k.startswith("infer_")}
        d["fall_pose_avg_infer_ms"] = (
            round(self.infer_ms_total / self.infer_n, 2) if self.infer_n else None)
        return d


class FallPoseAdapter:
    def __init__(self, enabled=None, model_factory=None, decision=None,
                 queue_size=None, interval_seconds=None, shadow=None,
                 pose=None):
        self.enabled = FALL_POSE_ENABLED if enabled is None else bool(enabled)
        self.shadow = SHADOW_MODE if shadow is None else bool(shadow)
        # Pose comes from the SHARED L960 runtime, which Behaviour also uses.
        # This adapter no longer owns an engine: one TensorRT context serves
        # both analytics instead of two identical ~10.4 GB contexts. When
        # Behaviour has already posed a frame this camera/frame_id, Fall gets
        # that result instead of inferring it again.
        #
        # Only the PROVIDER changed. The candidate gate, the association, the
        # state machine and every threshold are exactly as validated.
        self._pose = pose or pose_l960_runtime.shared_l960
        self._model_factory = model_factory    # tests only; None in production
        self._model = None
        self._model_failed = False          # latched: never retried in a hot loop
        self._model_lock = threading.Lock()
        self.policy = decision or FallPosePolicy(
            model_name=os.path.basename(POSE_MODEL), imgsz=POSE_IMGSZ)

        self._queue = queue.Queue(maxsize=queue_size or QUEUE_SIZE)
        self._results = queue.Queue()
        self._interval = (INTERVAL_SECONDS if interval_seconds is None
                          else interval_seconds)
        self._camera_lock = threading.Lock()
        self._enabled_cameras = frozenset()
        self._last_submit = {}
        self._hot = defaultdict(dict)       # camera -> track -> hot-until ts
        self._stats = _Stats()
        self._closed = False

        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._run, name="fall-pose-adapter", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------- arming
    def set_enabled_cameras(self, camera_ids):
        """Driven by the dashboard's existing `fall_detection` feature flag.

        No new flag, no new env var, and no camera is armed here that the
        operator has not armed in the dashboard.
        """
        wanted = frozenset(c.upper() for c in (camera_ids or ()))
        with self._camera_lock:
            changed = wanted != self._enabled_cameras
            self._enabled_cameras = wanted
        if changed:
            names = ", ".join(f"{c}=ON" for c in sorted(wanted)) or "no camera armed"
            print(f"[FALL-POSE] camera configuration: {names}"
                  + ("  (SHADOW MODE - no events will be raised)"
                     if self.shadow else ""))

    def camera_armed(self, camera_id):
        with self._camera_lock:
            return bool(camera_id) and camera_id.upper() in self._enabled_cameras

    # ------------------------------------------------------------ produce
    def submit(self, camera_id, frame, tracks, timestamp=None, frame_id=None):
        """Camera thread. Cheap geometry plus one put_nowait, or nothing."""
        if not self.enabled or self._closed or frame is None:
            return False
        if not self.camera_armed(camera_id):
            self._stats.skipped_not_armed += 1
            return False

        now = time.monotonic() if timestamp is None else timestamp
        hot = self._hot[camera_id]

        people = []
        for t in (tracks or ()):
            if getattr(t, "group", None) != PERSON_GROUP or not t.track_id:
                continue
            bbox = tuple(float(v) for v in t.bbox)
            if is_candidate(bbox):
                hot[t.track_id] = now + CANDIDATE_HOLD_SECONDS
                self._stats.candidates += 1
                people.append((t.track_id, bbox))
            elif hot.get(t.track_id, 0.0) > now:
                # Still hot: a dip in aspect ratio must not cut a confirmation
                # window short.
                people.append((t.track_id, bbox))
            else:
                self._stats.skipped_not_candidate += 1
        for tid in [k for k, until in hot.items() if until <= now]:
            del hot[tid]

        if not people:
            return False

        last = self._last_submit.get(camera_id)
        if self._interval > 0 and last is not None and (now - last) < self._interval:
            self._stats.skipped_interval += 1
            return False

        h, w = frame.shape[:2]
        try:
            self._queue.put_nowait({
                "camera_id": camera_id, "frame": frame.copy(),
                "tracks": people, "timestamp": now, "frame_id": frame_id,
                "frame_width": w, "frame_height": h,
            })
        except queue.Full:
            # Dropping is correct: a stale frame is worth less than a stalled
            # camera, and the fall is still there on the next one.
            self._stats.dropped_queue_full += 1
            return False
        self._last_submit[camera_id] = now
        self._stats.submitted += 1
        return True

    def flush(self):
        """Main thread. Confirmed FallPoseEvents, if any."""
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    # -------------------------------------------------------------- model
    def _ensure_model(self):
        """Return the pose provider, or None if it is unusable.

        In production this is the shared L960 runtime, which is lazy and owns
        its own latched-failure handling - so a broken engine turns Fall off
        once and quietly, exactly as before. `model_factory` is a test seam.
        """
        if self._model is not None:
            return self._model
        if self._model_failed:
            return None
        with self._model_lock:
            if self._model is not None or self._model_failed:
                return self._model
            if self._model_factory is not None:
                try:
                    self._model = self._model_factory()
                except Exception as exc:  # noqa: BLE001
                    self._model_failed = True
                    self._stats.model_errors += 1
                    print(f"[FALL-POSE] pose model unavailable, fall detection "
                          f"is OFF: {type(exc).__name__}: {exc}")
                return self._model
            if not self._pose.available():
                self._model_failed = True
                self._stats.model_errors += 1
                print("[FALL-POSE] shared L960 runtime unavailable, fall "
                      "detection is OFF")
                return None
            self._model = self._pose.acquire()
            return self._model

    # ------------------------------------------------------------- worker
    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                self._process(item)
            except Exception as exc:  # noqa: BLE001
                self._stats.worker_errors += 1
                print(f"[FALL-POSE] worker error (contained): "
                      f"{type(exc).__name__}: {exc}")
            finally:
                self._stats.processed += 1
                self._queue.task_done()

    def _process(self, item):
        model = self._ensure_model()
        if model is None:
            return

        t0 = time.perf_counter()
        boxes, kxy, kcf = self._infer(model, item["frame"],
                                      camera_id=item["camera_id"],
                                      frame_id=item.get("frame_id"))
        self._stats.infer_ms_total += (time.perf_counter() - t0) * 1000.0
        self._stats.infer_n += 1
        if boxes is None or kxy is None or not len(boxes):
            return
        self._stats.pose_persons += len(boxes)

        camera_id = item["camera_id"]
        now = item["timestamp"]
        # Section 9: associate pose to the EXISTING tracks by IoU, ONE-TO-ONE.
        #
        # Each pose detection belongs to exactly one person, so it may be
        # claimed by at most one track. Taking each track's best match
        # independently let two overlapping tracks consume the SAME detection
        # and each raise its own fall_detected - which happened in production
        # on 2026-09-08: EVT-35438406 (IoU 0.798) and EVT-B99D951E (IoU 0.325)
        # carried byte-identical torso/trunk/knee angles because both were
        # handed one person's keypoints. The weaker claimant's own box was
        # 84x203 - twice as tall as wide - so it was credited with a
        # horizontal torso it could not physically have had.
        #
        # Greedy best-IoU-wins: strongest pair first, then neither that track
        # nor that detection can be used again. Nothing else changes - the
        # IoU floor, the thresholds and the state machine are untouched.
        pairs = []
        for ti, (track_id, bbox) in enumerate(item["tracks"]):
            for i in range(len(boxes)):
                score = box_iou(bbox, boxes[i])
                if score >= POSE_MATCH_IOU:
                    pairs.append((score, ti, i))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

        assigned = {}                      # track index -> (pose index, iou)
        used_tracks, used_poses = set(), set()
        for score, ti, i in pairs:
            if ti in used_tracks or i in used_poses:
                self._stats.contested_poses += 1
                continue
            assigned[ti] = (i, score)
            used_tracks.add(ti)
            used_poses.add(i)

        for ti, (track_id, bbox) in enumerate(item["tracks"]):
            if ti not in assigned:
                self._stats.unassociated_tracks += 1
                continue
            bi, best = assigned[ti]

            result = self.policy.observe(
                camera_id, track_id, kxy[bi], kcf[bi], bbox,
                observed_at=now, pose_track_iou=best,
                frame_width=item["frame_width"], frame_height=item["frame_height"])
            if result.event is None:
                continue
            if self.shadow:
                self._stats.shadow_events += 1
                m = result.event.metadata()
                print(f"[FALL-POSE][SHADOW] would raise {EVENT_TYPE} on "
                      f"{camera_id} track {track_id}: torso={m['torso_angle']} "
                      f"deg for {m['pose_duration']}s, "
                      f"{m['confident_keypoints']} keypoints, iou={m['pose_track_iou']}")
                continue
            self._stats.events += 1
            # The decision frame, for the evidence crop. A reference to this
            # worker's own frame.copy() from submit() - no second copy, decode
            # or inference. handle_camera_event crops event.bbox out of it,
            # exactly as PPE does with observation.frame.
            result.event.frame = item["frame"]
            self._results.put(result.event)
        self.policy.prune(now)

    def _infer(self, model, frame, camera_id=None, frame_id=None):
        """One frame of pose, from the shared runtime.

        `classes=[0]` used to be passed here and is deliberately gone: this
        engine has exactly one class (person), so the filter was a no-op -
        verified by comparing both call shapes on a real frame, which produced
        bit-identical boxes, keypoints and confidences. Dropping it is what
        lets Behaviour's batched result and Fall's single-frame result be the
        same object.
        """
        if hasattr(model, "get_or_infer"):
            res = model.get_or_infer(camera_id, frame_id, frame)
        else:                                   # test seam: a bare YOLO model
            res = model.predict(source=[frame], conf=POSE_CONF,
                                imgsz=POSE_IMGSZ, verbose=False)[0]
        return unpack_pose(res)

    # ------------------------------------------------------------ shutdown
    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)
        s = self._stats.as_dict()
        print(f"[FALL-POSE] shutdown: submitted={s['fall_pose_submitted']} "
              f"processed={s['fall_pose_processed']} events={s['fall_pose_events']} "
              f"uncertain={self.policy.uncertain} "
              f"dropped={s['fall_pose_dropped_queue_full']} "
              f"errors={s['fall_pose_worker_errors'] + s['fall_pose_model_errors']}")

    def queue_depth(self):
        return self._queue.qsize()

    def stats(self):
        d = self._stats.as_dict()
        d.update(self.policy.stats())
        d["fall_pose_queue_depth"] = self.queue_depth()
        d["fall_pose_model_loaded"] = self._model is not None
        d["fall_pose_model_failed"] = self._model_failed
        d["fall_pose_shadow"] = self.shadow
        if self._pose is not None:
            d.update(self._pose.stats())
        return d


def _warm_cuda():
    """Establish a CUDA context before TensorRT builds its runtime.

    On this host, `createInferRuntime` fails with "CUDA initialization
    failure with error: 2" when no CUDA context exists yet - and TensorRT
    reports that in native code, aborting the process rather than raising
    something Python can catch. Inside multicam_inf.py torch has long since
    initialised CUDA, so this is a no-op there; it matters for any process
    that loads this adapter on its own, such as the replay harness.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.mem_get_info(0)
    except Exception as exc:  # noqa: BLE001
        print(f"[FALL-POSE] CUDA warm-up skipped: {type(exc).__name__}: {exc}")


def _load_pose_model():
    from ultralytics import YOLO
    _warm_cuda()
    if os.path.exists(POSE_MODEL):
        try:
            return YOLO(POSE_MODEL, task="pose")
        except Exception as exc:  # noqa: BLE001
            print(f"[FALL-POSE] engine load failed ({type(exc).__name__}: {exc}), "
                  f"falling back to {POSE_MODEL_FALLBACK}")
    return YOLO(POSE_MODEL_FALLBACK, task="pose")


def build_production_adapter():
    return FallPoseAdapter()
