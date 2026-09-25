"""
Fight (violence) detection - fully asynchronous adapter.

    adapter = build_production_adapter()              # once, at startup
    adapter.set_enabled_cameras(["CAM-R22"])          # from the dashboard
    adapter.submit(camera_id, frame, tracks, ts)      # per camera, per tick
    fights = adapter.flush()                          # once per tick

WHY AN ADAPTER AND NOT AN INLINE POLICY
---------------------------------------
Unlike fall_policy.py, this genuinely needs a model. The decision is entirely
keypoint-velocity based: a strike is a wrist or ankle moving faster than
STRIKE_SPEED_PS body-heights per second that lands within STRIKE_REACH of the
other person's torso. There is no bounding-box shortcut. So it follows
fire_smoke_adapter.py's shape instead - bounded queue, dedicated worker, lazy
model, drop rather than block - and the camera loop never waits for pose.

It does NOT add a person detector or a tracker. pose_runtime attaches keypoints
to the tracks multicam_inf.py already produced.

THE ONE DELIBERATE CHANGE FROM THE PROTOTYPE
--------------------------------------------
test_fightfall_det.py calls limb_velocities(..., TARGET_FPS) with a CONSTANT
fps, which is only correct because it throttles every source to exactly
1/TARGET_FPS. This adapter cannot: a bounded queue that drops under load has a
variable, unknown gap between the frames it actually processes. Multiplying a
larger real displacement by a constant fps OVERSTATES limb speed - so dropping
frames would manufacture strikes, and would do it hardest when the machine is
busiest. That is the wrong direction for a violence alarm.

So velocity uses the MEASURED interval between the two frames actually
differenced, and is discarded outright when that interval exceeds
FIGHT_MAX_GAP_SECONDS - a punch is over in well under a second, and
differencing across two seconds is meaningless however it is normalised. This
is the same correction fall_policy.py made when it re-clocked the prototype's
frame counts onto wall-clock seconds.

AND ONE MORE: THE COOLDOWN IS IN SECONDS, AND LONGER
----------------------------------------------------
The prototype's FIGHT_COOLDOWN_FRAMES = 15 is 0.5 s at 30 fps. Offline
validation showed one pair in fight3.mp4 re-firing five times at exactly 0.5 s
intervals. As an operator-facing event that is one fight reported ten times.
The window here is FIGHT_COOLDOWN_SECONDS, defaulting to 60 - the pipeline's
own debouncer is a second guard, but a sustained fight must not depend on it
alone (fall_policy.py makes the same argument).

Every threshold that decides WHAT a fight is - proximity, IOU, strike speed,
reach, mutual confirmation, minimum keypoints - is used verbatim from
test_fightfall_det.py. Only the two clocks above changed.
"""

import os
import queue
import threading
import time
from collections import defaultdict, deque


def _bool_env(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


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


# The dashboard feature that arms this. Both this key and the event type below
# ALREADY EXIST and are already mapped to each other in cameras/views.py's
# EVENT_FEATURE_KEY - this adapter adds no vocabulary.
FIGHT_CAMERA_FEATURE = "violence_detection"
EVENT_TYPE = "violence_detected"
PERSON_GROUP = "person"

FIGHT_ENABLED = _bool_env("FIGHT_ENABLED", True)      # kill switch only
FIGHT_QUEUE_SIZE = _int_env("FIGHT_QUEUE_SIZE", 8)
FIGHT_INTERVAL_SECONDS = _float_env("FIGHT_INTERVAL_SECONDS", 0.1)
FIGHT_SHUTDOWN_SECONDS = _float_env("FIGHT_SHUTDOWN_SECONDS", 5.0)

# Velocity clock (see module docstring).
FIGHT_MAX_GAP_SECONDS = _float_env("FIGHT_MAX_GAP_SECONDS", 0.5)
# Event debounce, in seconds rather than the prototype's frames.
FIGHT_COOLDOWN_SECONDS = _float_env("FIGHT_COOLDOWN_SECONDS", 60.0)
FIGHT_WINDOW_SECONDS = _float_env("FIGHT_WINDOW_SECONDS", 1.2)
FIGHT_TRACK_CACHE_CAP = _int_env("FIGHT_TRACK_CACHE_CAP", 4096)


class FightEvent:
    """One confirmed fight between two tracked people."""

    def __init__(self, camera_id, track_a, track_b, strikes_a, strikes_b,
                 bbox, observed_at, frame_width=None, frame_height=None):
        self.camera_id = camera_id
        self.track_a = track_a
        self.track_b = track_b
        self.strikes_a = strikes_a
        self.strikes_b = strikes_b
        self.bbox = bbox
        self.observed_at = observed_at
        self.frame_width = frame_width
        self.frame_height = frame_height

    @property
    def track_id(self):
        """The pipeline wants one track id on the event; report the first."""
        return self.track_a

    def metadata(self):
        return {
            "track_a": self.track_a,
            "track_b": self.track_b,
            "strikes_a": self.strikes_a,
            "strikes_b": self.strikes_b,
            "window_seconds": FIGHT_WINDOW_SECONDS,
            "mutual": True,
            "method": "pose_limb_velocity_strike",
            "pose_used": True,
        }

    def scope(self):
        """Debounce per PAIR, so two separate fights are two events."""
        a, b = sorted((str(self.track_a), str(self.track_b)))
        return f"fight:{a}:{b}"

    def __repr__(self):
        return (f"<FightEvent {self.camera_id} {self.track_a}~{self.track_b} "
                f"strikes={self.strikes_a}/{self.strikes_b}>")


class _Stats:
    def __init__(self):
        self.submitted = 0
        self.processed = 0
        self.dropped_queue_full = 0
        self.skipped_interval = 0
        self.raised = 0
        self.worker_errors = 0
        self.velocity_gaps_discarded = 0

    def as_dict(self):
        return dict(self.__dict__)


class FightAdapter:
    """Bounded queue in, confirmed fights out. Never blocks the camera loop."""

    def __init__(self, enabled=None, pose=None, logic=None, queue_size=None,
                 interval_seconds=None):
        self.enabled = FIGHT_ENABLED if enabled is None else bool(enabled)
        self._pose = pose
        self._logic = logic
        self._queue = queue.Queue(maxsize=queue_size or FIGHT_QUEUE_SIZE)
        self._interval = (FIGHT_INTERVAL_SECONDS if interval_seconds is None
                          else interval_seconds)
        self._results = queue.Queue()
        self._camera_lock = threading.Lock()
        self._enabled_cameras = frozenset()
        self._last_submit = {}
        self._stats = _Stats()
        self._closed = False
        self._acquired = False

        # per camera -> per track -> (kpts, confs, timestamp)
        self._prev = defaultdict(dict)
        # per camera -> per pair -> deque[(timestamp, a_hits, b_hits)]
        self._pair_hits = defaultdict(lambda: defaultdict(deque))
        self._cooldown = {}

        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._run, name="fight-adapter", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------ arming
    def set_enabled_cameras(self, camera_ids):
        wanted = frozenset(c.upper() for c in (camera_ids or ()))
        with self._camera_lock:
            changed = wanted != self._enabled_cameras
            self._enabled_cameras = wanted
        if changed:
            names = ", ".join(f"{c}=ON" for c in sorted(wanted)) or "no camera armed"
            print(f"[FIGHT] camera configuration: {names}")
        # Build the pose engine only once a camera actually wants it.
        if wanted and not self._acquired and self._pose is not None:
            self._acquired = True
            self._pose.acquire()

    def camera_armed(self, camera_id):
        with self._camera_lock:
            return bool(camera_id) and camera_id.upper() in self._enabled_cameras

    # ------------------------------------------------------------ produce
    def submit(self, camera_id, frame, tracks, timestamp=None, frame_id=None):
        """Called from the camera loop. Copies and returns; never waits."""
        if not self.enabled or self._closed or frame is None:
            return False
        if not self.camera_armed(camera_id):
            return False

        people = [(t.track_id, tuple(t.bbox)) for t in (tracks or ())
                  if getattr(t, "group", None) == PERSON_GROUP and t.track_id]
        # A fight needs two people. One person is not a cheap check to skip -
        # it avoids a pose inference that could not produce an event anyway.
        if len(people) < 2:
            return False

        now = time.monotonic() if timestamp is None else timestamp
        last = self._last_submit.get(camera_id)
        if last is not None and (now - last) < self._interval:
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
            # Dropping is correct: the camera loop must not wait on analytics.
            self._stats.dropped_queue_full += 1
            return False
        self._last_submit[camera_id] = now
        self._stats.submitted += 1
        return True

    def flush(self):
        """Called once per tick from the main loop. Drains confirmed fights."""
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                break
        return out

    # ------------------------------------------------------------- worker
    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                self._process(item)
            except Exception as exc:  # noqa: BLE001 - a worker fault is not fatal
                self._stats.worker_errors += 1
                print(f"[FIGHT] worker error (contained): "
                      f"{type(exc).__name__}: {exc}")
            finally:
                self._stats.processed += 1
                self._queue.task_done()

    def _process(self, item):
        if self._pose is None or not self._pose.available:
            return
        logic = self._logic
        if logic is None:
            return

        cam = item["camera_id"]
        now = item["timestamp"]
        # camera_id + frame_id let the shared runtime compute this frame ONCE
        # and hand the same result to every other consumer asking for it.
        people = self._pose.infer(item["frame"], item["tracks"],
                                  camera_id=cam, frame_id=item.get("frame_id"))
        if len(people) < 2:
            self._remember(cam, people, now)
            return

        limb_vel = self._velocities(cam, people, now, logic)
        self._remember(cam, people, now)

        for ta, tb, _ca, _cb, a_hits, b_hits in logic.detect_fights(people, limb_vel):
            key = tuple(sorted((str(ta), str(tb))))
            hist = self._pair_hits[cam][key]
            hist.append((now, a_hits, b_hits))
            while hist and (now - hist[0][0]) > FIGHT_WINDOW_SECONDS:
                hist.popleft()

            a_n = sum(1 for _, a, _ in hist if a)
            b_n = sum(1 for _, _, b in hist if b)
            if a_n < logic.FIGHT_MIN_STRIKES_EACH or b_n < logic.FIGHT_MIN_STRIKES_EACH:
                continue
            if now < self._cooldown.get((cam, key), 0.0):
                continue
            self._cooldown[(cam, key)] = now + FIGHT_COOLDOWN_SECONDS
            hist.clear()

            bbox = self._union_bbox(people, ta, tb)
            self._stats.raised += 1
            event = FightEvent(
                camera_id=cam, track_a=str(ta), track_b=str(tb),
                strikes_a=a_n, strikes_b=b_n, bbox=bbox, observed_at=now,
                frame_width=item["frame_width"], frame_height=item["frame_height"])
            # The decision frame, for the evidence crop: a reference to this
            # worker's own frame.copy(), the frame the strikes were measured on.
            event.frame = item["frame"]
            self._results.put(event)

    def _velocities(self, cam, people, now, logic):
        """Limb speeds using the MEASURED interval - see module docstring."""
        prev = self._prev[cam]
        vel = {}
        for tid, kpts, confs, bbox in people:
            entry = prev.get(tid)
            if entry is None:
                continue
            pk, pc, pt = entry
            dt = now - pt
            if dt <= 0 or dt > FIGHT_MAX_GAP_SECONDS:
                # Too long a gap: a strike would be over. Reporting a speed
                # here would be inventing one.
                self._stats.velocity_gaps_discarded += 1
                continue
            vel[tid] = logic.limb_velocities(pk, pc, kpts, confs, bbox, 1.0 / dt)
        return vel

    def _remember(self, cam, people, now):
        prev = self._prev[cam]
        for tid, kpts, confs, _bbox in people:
            prev[tid] = (kpts, confs, now)
        if len(prev) > FIGHT_TRACK_CACHE_CAP:
            for tid in sorted(prev, key=lambda t: prev[t][2])[:len(prev) - FIGHT_TRACK_CACHE_CAP]:
                del prev[tid]

    @staticmethod
    def _union_bbox(people, ta, tb):
        boxes = [p[3] for p in people if p[0] in (ta, tb)]
        if not boxes:
            return None
        x1 = min(float(b[0]) for b in boxes)
        y1 = min(float(b[1]) for b in boxes)
        x2 = max(float(b[2]) for b in boxes)
        y2 = max(float(b[3]) for b in boxes)
        return [int(x1), int(y1), int(x2), int(y2)]

    # ----------------------------------------------------------- shutdown
    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=FIGHT_SHUTDOWN_SECONDS)
        if self._pose is not None and self._acquired:
            self._pose.release()
        s = self._stats.as_dict()
        print(f"[FIGHT] shutdown  submitted={s['submitted']} processed={s['processed']} "
              f"queue_dropped={s['dropped_queue_full']} raised={s['raised']} "
              f"worker_errors={s['worker_errors']}")

    def queue_depth(self):
        return self._queue.qsize()

    def stats(self):
        out = self._stats.as_dict()
        out["queue_depth"] = self.queue_depth()
        with self._camera_lock:
            out["cameras"] = sorted(self._enabled_cameras)
        return out


def build_production_adapter():
    """Wire the real pose runtime and the prototype's own decision functions."""
    import pose_runtime
    import fight_fall_prototype as logic
    return FightAdapter(pose=pose_runtime.shared_pose, logic=logic)
