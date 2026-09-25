"""
Shared YOLO26L-pose @960 runtime - ONE engine, ONE TensorRT context, two
consumers: Behaviour and Fall.

WHY THIS EXISTS
---------------
Behaviour and Fall run the identical model at the identical settings
(yolo26l-pose.engine, imgsz 960, conf 0.25, COCO-17). Loaded separately they
each allocate their own TensorRT execution context of ~10.4 GB - measured, not
estimated - for the same 54 MB of weights. On this host that is the difference
between fitting in memory and not.

Sharing also removes a second inference whenever both analytics want the same
frame. That overlap is narrower than the memory saving: Behaviour is armed on
CAM-R12 and CAM-R22, Fall on CAM-R12 and CAM-R20, so today only CAM-R12
benefits from result sharing. The context saving applies always.

    Behaviour ---\
                  >--- shared L960 runtime (this module)
    Fall --------/

Fight and Fence deliberately do NOT use this. They stay on yolo26m-pose at 640
with conf 0.35 through pose_runtime.shared_pose, because changing Fight's pose
configuration was measured to change Fight's decisions (fight2.mp4: 3 events
-> 0). Two pose profiles is the correct architecture here, not one.

IDENTITY
--------
Results are keyed by `(camera_id, frame_id)` - never by frame_id alone, which
is per-camera and would collide across cameras immediately.

`frame_id` restarts at 0 when a CameraStream reconnects, so the key alone is
not sufficient either. Every entry therefore also carries a cheap content
fingerprint of the frame it came from, and a lookup whose fingerprint does not
match is treated as a miss. Combined with the TTL that makes a reconnect
unable to inherit a previous connection's result.

TWO CALL SHAPES, ONE ENGINE
---------------------------
Behaviour is synchronous and batched: it collects a frame per armed camera
during the tick and infers them together at flush. That is validated
behaviour and is preserved exactly - `infer_batch()` issues ONE predict call
for the whole chunk, the same call Behaviour used to make itself.

Fall is asynchronous and candidate-driven: its worker asks for one frame at a
time, and only for tracks its aspect-ratio gate selected. `get_or_infer()`
serves it from cache when Behaviour has already posed that frame, waits if
that inference is in flight, and otherwise infers it alone.

Neither path runs on a camera thread.
"""

import os
import threading
import time
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISTANCE_MEASURING = os.path.join(BASE_DIR, "distance_measuring")

#: Deliberately the same defaults Behaviour and Fall already validated. This
#: module is an inference PROVIDER; it owns no thresholds and no policy.
ENGINE_PATH = os.getenv("POSE_L960_ENGINE",
                        os.path.join(DISTANCE_MEASURING, "models",
                                     "yolo26l-pose.engine"))
MODEL_PATH = os.getenv("POSE_L960_MODEL_PT",
                       os.path.join(DISTANCE_MEASURING, "models",
                                    "yolo26l-pose.pt"))
IMG_SIZE = int(os.getenv("POSE_L960_IMGSZ", "960"))
POSE_CONF = float(os.getenv("POSE_L960_CONF", "0.25"))

CACHE_SIZE = int(os.getenv("POSE_L960_CACHE_SIZE", "64"))
CACHE_TTL_SECONDS = float(os.getenv("POSE_L960_CACHE_TTL_SECONDS", "2.0"))
#: How long a single-frame consumer will wait for an inference somebody else
#: already started before giving up and doing its own. Bounded so a stalled
#: batch can never pin a worker thread indefinitely.
WAIT_TIMEOUT_SECONDS = float(os.getenv("POSE_L960_WAIT_TIMEOUT", "1.0"))


def frame_fingerprint(frame):
    """Cheap content signature - a few hundred sampled pixels.

    Exists so a reconnect, which restarts frame_id at 0, cannot inherit the
    previous connection's result for the same (camera_id, frame_id).
    """
    if frame is None:
        return None
    try:
        return (frame.shape, int(frame[::64, ::64, 0].sum()))
    except Exception:  # noqa: BLE001
        return None


class _Entry:
    __slots__ = ("result", "published_at", "fingerprint")

    def __init__(self, result, published_at, fingerprint):
        self.result = result
        self.published_at = published_at
        self.fingerprint = fingerprint


class _Pending:
    """An inference somebody has committed to run, but has not finished."""
    __slots__ = ("event", "fingerprint", "reserved_at")

    def __init__(self, fingerprint, reserved_at):
        self.event = threading.Event()
        self.fingerprint = fingerprint
        self.reserved_at = reserved_at


class _Stats:
    def __init__(self):
        self.requests = 0
        self.inferences = 0
        self.frames_inferred = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.waits = 0
        self.wait_timeouts = 0
        self.evictions = 0
        self.expired = 0
        self.fingerprint_rejects = 0
        self.errors = 0
        self.batches = 0
        self.batch_frames_total = 0
        self.batch_max = 0
        self.infer_ms_total = 0.0

    def as_dict(self):
        return {
            "l960_requests": self.requests,
            "l960_inferences": self.inferences,
            "l960_frames_inferred": self.frames_inferred,
            "l960_cache_hits": self.cache_hits,
            "l960_cache_misses": self.cache_misses,
            "l960_waits": self.waits,
            "l960_wait_timeouts": self.wait_timeouts,
            "l960_evictions": self.evictions,
            "l960_expired": self.expired,
            "l960_fingerprint_rejects": self.fingerprint_rejects,
            "l960_errors": self.errors,
            "l960_batches": self.batches,
            "l960_batch_avg": (round(self.batch_frames_total / self.batches, 2)
                               if self.batches else None),
            "l960_batch_max": self.batch_max,
            "l960_avg_infer_ms": (round(self.infer_ms_total / self.inferences, 2)
                                  if self.inferences else None),
        }


class PoseL960Runtime:
    def __init__(self, model_factory=None):
        self._factory = model_factory or _load_l960_model
        self._model = None
        self._failed = False                 # latched
        self._model_lock = threading.Lock()
        self._lock = threading.Lock()        # guards cache + pending
        # Serialises the model call itself. A TensorRT execution context is
        # not thread-safe, and ultralytics keeps mutable per-call state on the
        # predictor, so exactly one thread may be inside predict() at a time.
        self._infer_lock = threading.RLock()
        self._cache = OrderedDict()
        self._pending = {}
        self._consumers = 0
        self._closed = False
        self._stats = _Stats()

    # ------------------------------------------------------------ lifecycle
    def acquire(self):
        """Register a consumer. Loads nothing - the model is lazy."""
        with self._model_lock:
            self._consumers += 1
        return self

    def available(self):
        return not self._failed and not self._closed

    def ensure_loaded(self):
        """Force the lazy load now and say whether it worked.

        Behaviour calls this when the dashboard first arms a camera, because
        its existing contract is that a broken engine is discovered at arm
        time - not silently at the first flush.
        """
        return self._ensure_model() is not None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._failed or self._closed:
            return None
        with self._model_lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                model = self._factory()
                # Force the predictor into existence HERE, while the model
                # lock is held.
                #
                # ultralytics assigns YOLO.predictor only AFTER a ~1 s setup,
                # so two threads whose first predict() overlaps both see None
                # and each build their own backend - which on TensorRT means a
                # SECOND ~9.4 GB execution context for the same weights. That
                # is exactly the duplication this module exists to remove, and
                # it happened in production: Behaviour's flush on the main
                # thread and Fall's worker raced on first use.
                _warm_predictor(model)
                self._model = model
                print(f"[POSE-L960] shared pose runtime ready: {ENGINE_PATH} "
                      f"imgsz={IMG_SIZE} conf={POSE_CONF} "
                      f"consumers={self._consumers}")
            except Exception as exc:  # noqa: BLE001
                self._failed = True
                self._stats.errors += 1
                print(f"[POSE-L960] shared pose runtime unavailable: "
                      f"{type(exc).__name__}: {exc}")
            return self._model

    # ---------------------------------------------------------------- cache
    def _lookup(self, key, fingerprint, now):
        """Caller holds the lock."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if now - entry.published_at > CACHE_TTL_SECONDS:
            del self._cache[key]
            self._stats.expired += 1
            return None
        if (fingerprint is not None and entry.fingerprint is not None
                and fingerprint != entry.fingerprint):
            # Same (camera, frame_id), different pixels: a reconnect restarted
            # the counter. Never serve this.
            del self._cache[key]
            self._stats.fingerprint_rejects += 1
            return None
        self._cache.move_to_end(key)
        return entry.result

    def _publish(self, key, result, fingerprint, now):
        """Caller holds the lock."""
        self._cache[key] = _Entry(result, now, fingerprint)
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    def _expire(self, now):
        """Caller holds the lock."""
        for key in [k for k, e in self._cache.items()
                    if now - e.published_at > CACHE_TTL_SECONDS]:
            del self._cache[key]
            self._stats.expired += 1

    def reset_camera(self, camera_id):
        """Drop everything cached for one camera - use on a known reconnect."""
        with self._lock:
            for key in [k for k in self._cache if k[0] == camera_id]:
                del self._cache[key]

    # --------------------------------------------------- Behaviour: batched
    def reserve_batch(self, items):
        """Announce frames a batch is about to infer, so a single-frame
        consumer waits for that result instead of starting a second inference.

        `items` is [(camera_id, frame_id, frame)]. Returns the keys reserved
        by THIS call; keys already cached or already pending are not
        re-reserved. Every reserved key must later be published or released -
        `infer_batch` does both on every path.
        """
        reserved = []
        now = time.monotonic()
        with self._lock:
            for camera_id, frame_id, frame in items:
                if frame_id is None:
                    continue
                key = (camera_id, frame_id)
                if key in self._pending or key in self._cache:
                    continue
                self._pending[key] = _Pending(frame_fingerprint(frame), now)
                reserved.append(key)
        return reserved

    def release(self, keys):
        """Release reservations without a result - wakes every waiter so it
        can fall back rather than block. Called on failure paths."""
        with self._lock:
            for key in keys or ():
                pending = self._pending.pop(key, None)
                if pending is not None:
                    pending.event.set()

    def infer_batch(self, items):
        """ONE predict call for the whole chunk. Behaviour's batching is
        preserved exactly - this issues the same call Behaviour used to make.

        `items` is [(camera_id, frame_id, frame)]. Returns the ultralytics
        Results in the same order, or None if the model is unavailable or the
        call raised. Each result is published under its (camera_id, frame_id)
        so Fall can consume it instead of inferring the frame again.
        """
        if not items:
            return []
        self._stats.requests += len(items)
        model = self._ensure_model()
        if model is None:
            self.release([(c, f) for c, f, _ in items if f is not None])
            return None

        keys = [((c, f) if f is not None else None) for c, f, _ in items]
        frames = [frame for _, _, frame in items]
        fps = [frame_fingerprint(frame) for frame in frames]

        try:
            started = time.perf_counter()
            with self._infer_lock:
                results = model.predict(source=frames, imgsz=IMG_SIZE,
                                        conf=POSE_CONF, device=0, verbose=False)
            elapsed = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            self._stats.errors += 1
            # Requirement: a failed inference must release every waiter.
            self.release([k for k in keys if k is not None])
            print(f"[POSE-L960] batch inference failed (contained): "
                  f"{type(exc).__name__}: {exc}")
            return None

        now = time.monotonic()
        self._stats.inferences += 1
        self._stats.frames_inferred += len(frames)
        self._stats.infer_ms_total += elapsed
        self._stats.batches += 1
        self._stats.batch_frames_total += len(frames)
        self._stats.batch_max = max(self._stats.batch_max, len(frames))

        with self._lock:
            for key, result, fingerprint in zip(keys, results, fps):
                if key is None:
                    continue
                self._publish(key, result, fingerprint, now)
                pending = self._pending.pop(key, None)
                if pending is not None:
                    pending.event.set()
            self._expire(now)
        return list(results)

    # ------------------------------------------------- Fall: one frame, lazy
    def get(self, camera_id, frame_id, frame=None):
        """Cache lookup only. Never infers, never waits."""
        if frame_id is None:
            return None
        with self._lock:
            return self._lookup((camera_id, frame_id), frame_fingerprint(frame),
                                time.monotonic())

    def get_or_infer(self, camera_id, frame_id, frame,
                     timeout=WAIT_TIMEOUT_SECONDS):
        """Serve one frame: cache hit, wait for an in-flight inference, or
        infer it alone. Returns one ultralytics Result, or None.

        Never called from a camera thread - Fall's worker owns this.
        """
        self._stats.requests += 1
        fingerprint = frame_fingerprint(frame)
        key = (camera_id, frame_id) if frame_id is not None else None
        now = time.monotonic()

        if key is not None:
            with self._lock:
                hit = self._lookup(key, fingerprint, now)
                if hit is not None:
                    self._stats.cache_hits += 1
                    return hit
                self._stats.cache_misses += 1
                pending = self._pending.get(key)
                if pending is None:
                    # Commit to running it, so a concurrent asker waits for us
                    # rather than starting a third inference.
                    self._pending[key] = _Pending(fingerprint, now)
                    pending = None
                else:
                    self._stats.waits += 1
            if pending is not None:
                # Somebody else - typically Behaviour's pending batch - is
                # already going to produce this frame.
                if pending.event.wait(timeout):
                    with self._lock:
                        hit = self._lookup(key, fingerprint, time.monotonic())
                    if hit is not None:
                        self._stats.cache_hits += 1
                        return hit
                else:
                    self._stats.wait_timeouts += 1
                # Released without a result, or timed out: fall through and do
                # it ourselves rather than returning nothing.
                with self._lock:
                    if key not in self._pending:
                        self._pending[key] = _Pending(fingerprint, time.monotonic())
                    else:
                        key = None          # someone still owns it; don't publish
        else:
            self._stats.cache_misses += 1

        model = self._ensure_model()
        if model is None:
            if key is not None:
                self.release([key])
            return None
        try:
            started = time.perf_counter()
            with self._infer_lock:
                result = model.predict(source=[frame], imgsz=IMG_SIZE,
                                       conf=POSE_CONF, device=0, verbose=False)[0]
            elapsed = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            self._stats.errors += 1
            if key is not None:
                self.release([key])
            print(f"[POSE-L960] inference failed (contained): "
                  f"{type(exc).__name__}: {exc}")
            return None

        self._stats.inferences += 1
        self._stats.frames_inferred += 1
        self._stats.infer_ms_total += elapsed
        if key is not None:
            now = time.monotonic()
            with self._lock:
                self._publish(key, result, fingerprint, now)
                p = self._pending.pop(key, None)
                if p is not None:
                    p.event.set()
                self._expire(now)
        return result

    # ------------------------------------------------------------- shutdown
    def close(self):
        """Bounded: wakes every waiter, drops the cache, releases the model."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            for pending in self._pending.values():
                pending.event.set()
            self._pending.clear()
            self._cache.clear()
        self._model = None
        s = self._stats.as_dict()
        print(f"[POSE-L960] shutdown: requests={s['l960_requests']} "
              f"inferences={s['l960_inferences']} "
              f"cache_hits={s['l960_cache_hits']} errors={s['l960_errors']}")

    def stats(self):
        d = self._stats.as_dict()
        with self._lock:
            d["l960_cached"] = len(self._cache)
            d["l960_pending"] = len(self._pending)
        d["l960_model_loaded"] = self._model is not None
        d["l960_model_failed"] = self._failed
        d["l960_consumers"] = self._consumers
        return d


def _warm_predictor(model):
    """One throwaway inference so the ultralytics predictor and its TensorRT
    context exist before any second thread can call predict()."""
    import numpy as np
    try:
        model.predict(source=[np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)],
                      imgsz=IMG_SIZE, conf=POSE_CONF, device=0, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[POSE-L960] warm-up inference failed: "
              f"{type(exc).__name__}: {exc}")
        raise


def _load_l960_model():
    from ultralytics import YOLO
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.mem_get_info(0)
    except Exception as exc:  # noqa: BLE001
        print(f"[POSE-L960] CUDA warm-up skipped: {type(exc).__name__}: {exc}")
    if os.path.exists(ENGINE_PATH):
        try:
            return YOLO(ENGINE_PATH, task="pose")
        except Exception as exc:  # noqa: BLE001
            print(f"[POSE-L960] engine load failed ({type(exc).__name__}: {exc})")
            raise
    return YOLO(MODEL_PATH, task="pose")


#: The one instance. Both consumers import this, nobody constructs another.
shared_l960 = PoseL960Runtime()
