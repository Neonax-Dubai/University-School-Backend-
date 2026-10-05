"""
Sleeping in class (Zayed) - SLEEPING_DETECTED.

Pose, not boxes: a seated student's box barely changes between sitting up and slumping over the
desk. Every SLEEP_SAMPLE_SECONDS per armed camera one frame is handed to a worker thread, which
runs the SHARED YOLO26L-pose @960 runtime (pose_l960_runtime: the same engine and frame cache
Fall uses, so a frame both need is inferred once) and matches skeletons to the pipeline's person
tracks by IoU. Nothing here runs on the inference thread except a time check and a queue put.

One sample is HEAD-DOWN when, from COCO-17 keypoints:
  * both shoulders are confident (>= KPT_CONF) and at least MIN_SHOULDER_PX apart, and
  * the highest confident head point (nose / eyes / ears) rises less than HEAD_DROP_RATIO
    shoulder-widths above the shoulder line (upright, the head sits ~0.5-0.9 widths above it,
    less on a steep ceiling camera), or - no head point confident, the face buried in the arms -
    the top of the person's box rises less than that above the shoulders.
A track is SLEEPING when, over the last SLEEP_SECONDS, at least SLEEP_MIN_SAMPLES samples could
be judged, at least HEAD_DOWN_FRACTION of them were head-down, and the box centre stayed within
STILL_FRACTION of the box height (still: not writing, not reaching). One event per track per
SLEEP_COOLDOWN_SECONDS, with the student's crop as evidence.

HEURISTIC - pending validation on the real C101 cameras. The expected false positive is a student
reading with the head bowed low for minutes, which is why SLEEP_SECONDS is long.
"""
import os
import queue
import threading
import time
from collections import deque

import numpy as np

SLEEP_FEATURE = "sleeping_detection"
EVENT_TYPE = "SLEEPING_DETECTED"

SLEEP_SAMPLE_SECONDS = float(os.getenv("SLEEP_SAMPLE_SECONDS", "2.0"))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "60"))
SLEEP_MIN_SAMPLES = int(os.getenv("SLEEP_MIN_SAMPLES", "10"))
HEAD_DOWN_FRACTION = float(os.getenv("SLEEP_HEAD_DOWN_FRACTION", "0.8"))
# HEAD_DROP_RATIO: head must rise LESS than this many reference-widths above
# the shoulder line to count as HEAD-DOWN.
# Calibrated from debug logs on overhead CCTV:
#   sleeping frames → rise = 0.36–0.73   (single-shoulder bbox scale)
#   upright frames  → rise = 0.75–0.95   (both-shoulder inter-distance scale)
# Threshold of 0.72 sits cleanly in the gap between the two clusters.
#
# FOR DESK-LEVEL CAMERAS (not overhead): normal sitting gives rise=0.5-0.7,
# real sleep gives rise=0.3 down to NEGATIVE (head below shoulders).
# Lowered to 0.35 to require head near/below shoulder level.
HEAD_DROP_RATIO = float(os.getenv("SLEEP_HEAD_DROP_RATIO", "0.35"))
# Minimum rise to consider as potential sleep (filters out normal sitting).
# Normal sitting: rise=0.5-0.7. Real sleep: rise<0.4, often negative.
MIN_SLEEP_RISE = float(os.getenv("SLEEP_MIN_RISE", "0.4"))
# STILL_FRACTION: max centroid displacement over the window, as a fraction of box height.
# 0.15 (restored 2026-10-05). 0.5 let a student STANDING UP pass as still (0.40 measured on the
# camera_01 2026-09-26 17:25 production replay) and lets writing / reaching (0.2) through. The lean
# onto the desk only delays a real sleeper: once he has been still for a window, movement is ~0.
STILL_FRACTION = float(os.getenv("SLEEP_STILL_FRACTION", "0.15"))
SLEEP_COOLDOWN_SECONDS = float(os.getenv("SLEEP_COOLDOWN_SECONDS", "600"))
KPT_CONF = float(os.getenv("SLEEP_KPT_CONF", "0.4"))
MIN_SHOULDER_PX = float(os.getenv("SLEEP_MIN_SHOULDER_PX", "12"))
# MATCH_IOU: minimum IoU between main tracker bbox and pose model bbox to associate them.
# Raised from 0.3 to 0.45 to prevent wrong associations when detector and pose model
# detect different nearby people. At 0.3, two people standing close can be mismatched.
MATCH_IOU = float(os.getenv("SLEEP_MATCH_IOU", "0.45"))
# POSE_KEYPOINT_IOU: minimum fraction of pose keypoints (head/shoulders) that must fall
# inside the tracker bbox to validate a match. Prevents bbox-only matches on wrong person.
POSE_KEYPOINT_IOU = float(os.getenv("SLEEP_POSE_KPT_IOU", "0.3"))
SLEEP_DEBUG = os.getenv("SLEEP_DEBUG", "0") == "1"
QUEUE_SIZE = 6
TRACK_FORGET_SECONDS = 120.0
# IoU threshold for pose-box continuity tracker (lower than MATCH_IOU so
# we keep track even when a student shifts slightly between samples).
POSE_TRACK_IOU = float(os.getenv("SLEEP_POSE_TRACK_IOU", "0.20"))
# Time window to reconcile pose tracker IDs with main tracker IDs (seconds).
# When main detector re-detects a person previously only seen by pose model,
# we can link the POSE_* ID to the main tracker ID within this window.
POSE_RECONCILE_SECONDS = float(os.getenv("SLEEP_POSE_RECONCILE_SECONDS", "10.0"))

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6
HEAD_POINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


def head_down(kpts, confs, bbox):
    """(True|False|None, detail) for one skeleton. None = cannot be judged from this view.

    Supports single-shoulder detection: when a student sleeps with their head on
    a desk, one shoulder is often buried under the body and its keypoint confidence
    drops below KPT_CONF. We accept a single visible shoulder (using bbox width as
    the reference scale) so sleeping frames are still judged rather than discarded.

    Priority:
      1. Both shoulders confident  → inter-shoulder distance as scale
      2. One shoulder confident    → that shoulder's y; bbox_w * 0.5 as scale
      3. Neither confident         → return None (cannot judge)
    """
    ls_ok = confs[L_SHOULDER] >= KPT_CONF
    rs_ok = confs[R_SHOULDER] >= KPT_CONF
    bbox_w = max(1.0, float(bbox[2] - bbox[0]))

    if ls_ok and rs_ok:
        # Normal case: both shoulders visible
        sy = (kpts[L_SHOULDER][1] + kpts[R_SHOULDER][1]) / 2.0
        width = abs(kpts[L_SHOULDER][0] - kpts[R_SHOULDER][0])
        overhead_fallback = False
        single_shoulder = False
        if width < MIN_SHOULDER_PX:
            width = bbox_w * 0.5
            overhead_fallback = True
    elif ls_ok:
        # Right shoulder occluded (e.g. student leaning right onto desk)
        sy = kpts[L_SHOULDER][1]
        width = bbox_w * 0.5
        overhead_fallback = True
        single_shoulder = "left_only"
    elif rs_ok:
        # Left shoulder occluded (most common: student leaning left/forward onto desk)
        sy = kpts[R_SHOULDER][1]
        width = bbox_w * 0.5
        overhead_fallback = True
        single_shoulder = "right_only"
    else:
        return None, {"reason": "shoulders_not_visible",
                      "ls_conf": round(float(confs[L_SHOULDER]), 2),
                      "rs_conf": round(float(confs[R_SHOULDER]), 2)}

    head = [i for i in HEAD_POINTS if confs[i] >= KPT_CONF]
    detail_base = {
        "overhead_fallback": overhead_fallback,
        "single_shoulder": single_shoulder if (ls_ok ^ rs_ok) else False,
        "ls_conf": round(float(confs[L_SHOULDER]), 2),
        "rs_conf": round(float(confs[R_SHOULDER]), 2),
    }
    if head:
        rise = (sy - min(kpts[i][1] for i in head)) / width
        # Require rise < HEAD_DROP_RATIO (stricter threshold for desk-level cameras)
        # Also require rise < MIN_SLEEP_RISE to filter normal sitting (rise=0.5-0.7)
        is_down = rise < HEAD_DROP_RATIO and rise < MIN_SLEEP_RISE
        return is_down, {"head_rise": round(float(rise), 2), **detail_base}
    # No head keypoints confident — face buried in arms; use bbox top as proxy
    rise = (sy - bbox[1]) / width
    is_down = rise < HEAD_DROP_RATIO and rise < MIN_SLEEP_RISE
    return is_down, {"head_rise": round(float(rise), 2),
                      "face_hidden": True, **detail_base}


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
def match_skeletons(tracks, boxes, kxy, kcf):
    """[(track_id, bbox)] x pose boxes -> [(track_id, bbox, kpts, confs)], one skeleton per track."""
    out, used = [], set()
    for track_id, bbox in tracks:
        best, best_iou = None, MATCH_IOU
        for i, box in enumerate(boxes):
            if i in used:
                continue
            iou = _iou(bbox, box)
            if iou >= best_iou:
                best, best_iou = i, iou
        if best is not None:
            used.add(best)
            out.append((track_id, bbox, kxy[best], kcf[best]))
    return out


class SleepFinding:
    def __init__(self, camera_id, track_id, bbox, frame, frame_width, frame_height, observed_at,
                 samples, down_fraction, movement, span):
        self.camera_id = camera_id
        self.track_id = track_id
        self.bbox = [int(v) for v in bbox]
        self.frame = frame
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.observed_at = observed_at
        self.samples = samples
        self.down_fraction = down_fraction
        self.movement = movement
        self.span = span

    def scope(self):
        return f"sleep:{self.track_id}"

    def metadata(self):
        return {"track_id": self.track_id, "samples": self.samples,
                "head_down_fraction": round(self.down_fraction, 2),
                "movement_fraction": round(self.movement, 3), "duration_seconds": round(self.span, 1),
                "method": "pose_head_at_shoulder_level_and_still",
                "semantic_note": "head down and still for a sustained period - posture heuristic, not proof of sleep"}


class SleepTracker:
    """Pure decision state - no threads, no model - so it is testable on synthetic samples."""

    def __init__(self):
        self._samples = {}       # (camera, track) -> deque[(t, down|None, cx, cy, h)]
        self._last_event = {}
        self._last_seen = {}

    def observe(self, camera_id, track_id, bbox, down, now):
        key = (camera_id, track_id)
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        samples = self._samples.setdefault(key, deque())
        samples.append((now, down, cx, cy, max(1.0, bbox[3] - bbox[1])))
        self._last_seen[key] = now
        while samples and now - samples[0][0] > SLEEP_SECONDS:
            samples.popleft()
        return self._decide(key, samples, now)

    def _decide(self, key, samples, now):
        if not samples or now - samples[0][0] < SLEEP_SECONDS * 0.9:
            if SLEEP_DEBUG:
                span = now - samples[0][0] if samples else 0
                need = SLEEP_SECONDS * 0.9
                print(f"[SLEEP-DECIDE] {key}  BLOCK: window too short  "
                      f"span={span:.1f}s / need={need:.1f}s")
            return None                                   # not observed long enough yet
        judged = [s for s in samples if s[1] is not None]
        if len(judged) < SLEEP_MIN_SAMPLES:
            if SLEEP_DEBUG:
                print(f"[SLEEP-DECIDE] {key}  BLOCK: too few samples  "
                      f"judged={len(judged)} / need={SLEEP_MIN_SAMPLES}")
            return None
        down = sum(1 for s in judged if s[1]) / len(judged)
        if down < HEAD_DOWN_FRACTION:
            if SLEEP_DEBUG:
                print(f"[SLEEP-DECIDE] {key}  BLOCK: head-down frac too low  "
                      f"frac={down:.2f} / need={HEAD_DOWN_FRACTION}")
            return None
        cx = np.median([s[2] for s in samples])
        cy = np.median([s[3] for s in samples])
        h = float(np.median([s[4] for s in samples]))
        movement = max(float(np.hypot(s[2] - cx, s[3] - cy)) for s in samples) / h
        if movement > STILL_FRACTION:
            if SLEEP_DEBUG:
                print(f"[SLEEP-DECIDE] {key}  BLOCK: too much movement  "
                      f"movement={movement:.3f} / limit={STILL_FRACTION}")
            return None
        last = self._last_event.get(key)
        if last is not None and now - last < SLEEP_COOLDOWN_SECONDS:
            if SLEEP_DEBUG:
                print(f"[SLEEP-DECIDE] {key}  BLOCK: cooldown  "
                      f"elapsed={now-last:.0f}s / cooldown={SLEEP_COOLDOWN_SECONDS:.0f}s")
            return None
        self._last_event[key] = now
        span = samples[-1][0] - samples[0][0]
        samples.clear()
        if SLEEP_DEBUG:
            print(f"[SLEEP-DECIDE] {key}  *** SLEEPING TRIGGERED ***  "
                  f"down={down:.2f}  movement={movement:.3f}  span={span:.1f}s")
        return {"samples": len(judged), "down_fraction": down, "movement": movement, "span": span}

    def forget_camera(self, camera_id):
        for store in (self._samples, self._last_event, self._last_seen):
            for key in [k for k in store if k[0] == camera_id]:
                del store[key]

    def prune(self, now):
        for key in [k for k, t in self._last_seen.items() if now - t > TRACK_FORGET_SECONDS]:
            self._last_seen.pop(key, None)
            self._samples.pop(key, None)
        for key in [k for k, t in self._last_event.items() if now - t > SLEEP_COOLDOWN_SECONDS]:
            del self._last_event[key]


class SleepingAdapter:
    """submit() on the inference thread (cheap), pose + decisions on one worker, flush() back."""

    def __init__(self, pose=None, log=print):
        if pose is None:
            import pose_l960_runtime
            pose = pose_l960_runtime.shared_l960
        self._pose = pose
        self._log = log
        self._enabled = frozenset()
        self._lock = threading.Lock()
        self._next_sample = {}
        self._queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._results = queue.Queue(maxsize=64)
        self._tracker = SleepTracker()
        self._stop = threading.Event()
        self._thread = None
        self._acquired = False
        self.submitted = self.dropped = self.processed = self.errors = self.raised = self.no_pose = 0
        self.judged = self.head_down_samples = 0

    def set_enabled_cameras(self, camera_ids):
        wanted = frozenset(camera_ids or ())
        with self._lock:
            self._enabled = wanted
        if wanted and not self._acquired:
            self._pose.acquire()
            self._acquired = True
        if wanted and (self._thread is None or not self._thread.is_alive()):
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="sleeping", daemon=True)
            self._thread.start()

    def submit(self, camera_id, frame, tracks, observed_at, frame_id, frame_width, frame_height, simulated_now=None):
        if camera_id not in self._enabled or not tracks:
            return False
        now = simulated_now if simulated_now is not None else time.monotonic()
        if now < self._next_sample.get(camera_id, 0.0):
            return False
        self._next_sample[camera_id] = now + SLEEP_SAMPLE_SECONDS
        item = (camera_id, frame, [(t.track_id, list(t.bbox)) for t in tracks], observed_at, frame_id,
                frame_width, frame_height)
        try:
            self._queue.put_nowait(item)
            self.submitted += 1
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(*item)
            except Exception as exc:                      # noqa: BLE001 - never kill the worker
                self.errors += 1
                if self.errors <= 3 or self.errors % 100 == 0:
                    self._log(f"[SLEEPING] error (contained): {type(exc).__name__}: {exc}")

    def _process(self, camera_id, frame, tracks, observed_at, frame_id, frame_width, frame_height):
        result = self._pose.get_or_infer(camera_id, frame_id, frame)
        self.processed += 1
        if result is None or result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            self.no_pose += 1
            return
        boxes = result.boxes.xyxy.cpu().numpy()
        kxy = result.keypoints.xy.cpu().numpy()
        kcf = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
               else np.ones(kxy.shape[:2]))

        # ── Pass 1: main-tracker persons matched to pose skeletons ────────────
        # Uses the main detector's track ID so the alert bbox matches the
        # tracker's bounding box (the one drawn on screen). This is the
        # preferred path when the detector sees the person clearly.
        matched_pose_indices = set()
        for track_id, bbox in tracks:
            best, best_iou = None, MATCH_IOU
            for i, box in enumerate(boxes):
                if i in matched_pose_indices:
                    continue
                iou = _iou(bbox, box)
                if iou >= best_iou:
                    best, best_iou = i, iou
            if best is None:
                continue
            matched_pose_indices.add(best)
            kpts  = kxy[best]
            confs = kcf[best]
            down, _ = head_down(kpts, confs, bbox)
            if down is not None:
                self.judged += 1
                self.head_down_samples += int(down)
            decision = self._tracker.observe(camera_id, track_id, bbox, down, observed_at)
            if decision:
                finding = SleepFinding(camera_id, track_id, bbox, frame, frame_width, frame_height,
                                       observed_at, decision["samples"], decision["down_fraction"],
                                       decision["movement"], decision["span"])
                try:
                    self._results.put_nowait(finding)
                    self.raised += 1
                except queue.Full:
                    self.dropped += 1

        self._tracker.prune(observed_at)

    def flush(self):
        findings = []
        while True:
            try:
                findings.append(self._results.get_nowait())
            except queue.Empty:
                return findings

    def forget_camera(self, camera_id):
        self._next_sample.pop(camera_id, None)
        self._tracker.forget_camera(camera_id)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        self._log(f"[SLEEPING] shutdown submitted={self.submitted} processed={self.processed} "
                  f"judged={self.judged} head_down={self.head_down_samples} raised={self.raised} "
                  f"dropped={self.dropped} errors={self.errors}")

    def stats(self):
        return {"cameras": sorted(self._enabled), "submitted": self.submitted, "processed": self.processed,
                "judged": self.judged, "head_down_samples": self.head_down_samples, "no_pose": self.no_pose,
                "raised": self.raised, "dropped": self.dropped, "errors": self.errors,
                "queue_depth": self._queue.qsize()}

