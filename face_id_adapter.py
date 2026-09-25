"""
Face-ID adapter - the isolation boundary between the real-time CCTV
inference pipeline and everything Face-ID does.

    adapter = FaceIDAdapter()                                   # once, at startup
    ...
    adapter.submit(camera_id, tracked_objects, frame, ts)       # per camera, per tick
    ...
    adapter.close()                                             # at shutdown

WHY THIS EXISTS, AND WHAT IT DELIBERATELY DOES NOT DO
-----------------------------------------------------
The existing adapters in this pipeline (reid_adapter.py, ppe_adapter.py,
behaviour_adapter.py) queue crops in process_tracks() and then run their
model work SYNCHRONOUSLY, on the main inference thread, inside flush().
That is a sound design for ONE batched TensorRT forward pass.

Face-ID is not one forward pass. A single observation costs face detection
+ embedding + age/gender + a Qdrant search + a Qdrant write - measured at
~6.4 ms median / ~12.2 ms p99 for the model half alone, before any network
round-trip. Putting that on the inference thread would make the pipeline's
frame rate a function of face-model and Qdrant latency, which is exactly
the coupling this class exists to prevent.

So this adapter keeps the same SHAPE as its siblings (a cheap call in the
per-camera loop) but breaks the pattern in one deliberate way: it has no
flush() that does model work. submit() enqueues and returns; ONE background
worker owns everything expensive. The inference thread never waits on
Face-ID, never imports a face model, and cannot be made slower by a slow or
broken Qdrant.

    inference thread:  submit() -> cooldown check -> crop.copy() ->
                       put_nowait() -> return.  No model. No network.
    worker thread:     drains the queue, one item at a time, and calls the
                       injected processor.

THE PROCESSOR SEAM
------------------
This module imports NO face model, NO torch, NO onnxruntime and NO Qdrant
client - deliberately, and it should stay that way. The worker calls an
injected `processor` object with a .process(observation) method. Phase 1
ships _NullProcessor (counts and discards); Phase 2 injects
face_id_manager.FaceIDManager.

That seam is not decoration. It is what makes this file unit-testable on a
machine with no GPU, and it is what makes failure isolation REAL rather
than asserted: a test can inject a processor that raises on every single
item and prove the producer never notices.

ASYNCHRONOUS IS NOT FREE
------------------------
The queue guarantees the inference thread never BLOCKS on Face-ID. It does
not guarantee zero impact: the face models run on the same GPU as the YOLO
TensorRT engine, so the two contend for it. Nothing in this file should be
read as a claim of zero performance cost - measuring that contention is
Phase 5's job.

Env (all optional; defaults are the PoC values):
  FACE_ID_ENABLED                   master switch (default false)
  FACE_ID_QUEUE_SIZE                bounded queue depth (default 64)
  FACE_ID_COOLDOWN_SECONDS          per-local-track submit throttle (default 5.0)
  FACE_ID_SHUTDOWN_FLUSH_SECONDS    bound on close()'s join (default 5.0)
  FACE_ID_STATS_INTERVAL_SECONDS    [FACE-ID] counter print cadence (default 30.0)
  FACE_ID_MIN_PERSON_CROP_WIDTH     cheap sanity floor, px (default 32)
  FACE_ID_MIN_PERSON_CROP_HEIGHT    cheap sanity floor, px (default 64)
  FACE_ID_TRACK_CACHE_CAP           cooldown-dict cap (default 4096)

Camera arming: for the PoC the caller passes enabled_cameras explicitly (or
None for "every camera"). Dashboard-driven arming, mirroring how
reid_adapter.py reads Camera.reid_enabled, is Phase 4 - it is deliberately
NOT invented here.
"""
import os
import queue
import threading
import time


def _bool_env(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ============================================================
# CONFIGURATION
# ============================================================

#: EMERGENCY KILL SWITCH ONLY - not the way to turn Face-ID on for a camera.
#:
#: Defaults to True because the DASHBOARD decides which cameras run Face-ID,
#: through each camera's "Face Detection" AI-analytics toggle, and every
#: camera ships with that toggle OFF. So a pipeline started with no
#: environment configuration at all runs no Face-ID until an operator arms a
#: camera - the switch being on costs nothing.
#:
#: Setting this to 0 disables Face-ID for the whole process REGARDLESS of the
#: dashboard, and says so loudly at startup, so the state "dashboard says ON
#: but nothing happens" can never be silent.
FACE_ID_ENABLED = _bool_env("FACE_ID_ENABLED", True)

#: Bounded, and deliberately small. This queue is a shock absorber for a
#: brief burst, NOT a backlog: an observation that waits seconds for the
#: worker describes a person who has probably already left frame, so
#: dropping it costs less than the memory to hold it. Every entry pins a
#: person-crop ndarray, so depth is measured in megabytes, not items.
FACE_ID_QUEUE_SIZE = int(os.getenv("FACE_ID_QUEUE_SIZE", "64"))

#: Per (camera_id, local_track_id) submit throttle. The brief's "do not
#: process every frame" rail, and the single biggest lever on Face-ID's
#: total cost: a person standing in view for a minute submits ~12 times at
#: the default, not ~600. Matches the 5.0s the test_face.py prototype
#: validated as its own refresh cooldown.
FACE_ID_COOLDOWN_SECONDS = float(os.getenv("FACE_ID_COOLDOWN_SECONDS", "5.0"))

#: Upper bound on how long close() will wait for the worker to finish its
#: current item and exit. Shutdown must be bounded even if the processor is
#: wedged on a network call - see close().
FACE_ID_SHUTDOWN_FLUSH_SECONDS = float(os.getenv("FACE_ID_SHUTDOWN_FLUSH_SECONDS", "5.0"))

FACE_ID_STATS_INTERVAL_SECONDS = float(os.getenv("FACE_ID_STATS_INTERVAL_SECONDS", "30.0"))

#: How often each ARMED camera tells the dashboard that recognition is alive
#: on it, whether or not anybody is in front of it.
#:
#: The dashboard's absence sweep refuses to raise an absence alert for a
#: camera whose recognition cannot be shown to be working
#: (investigation/presence.py, recognition_is_healthy - 180s staleness). The
#: heartbeat used to be sent only by a completed FACE lookup, so an empty
#: room - the one situation an absence alert exists for - produced no faces,
#: no heartbeat, and an alert held for ever. A tick per camera per interval
#: reports the thing that is actually being asserted: frames from this camera
#: are reaching an armed face pipeline.
FACE_ID_HEARTBEAT_SECONDS = float(os.getenv("FACE_ID_HEARTBEAT_SECONDS", "30.0"))

#: Cheap structural sanity on the PERSON crop - not the face quality gate.
#: The real quality decision (face size, detection score, sharpness) needs
#: the face bbox, which only exists after detection, so it belongs to the
#: worker-side manager in Phase 2. This only rejects crops too small to
#: contain a resolvable face at all, so the queue is not spent on them.
FACE_ID_MIN_PERSON_CROP_WIDTH = int(os.getenv("FACE_ID_MIN_PERSON_CROP_WIDTH", "32"))
FACE_ID_MIN_PERSON_CROP_HEIGHT = int(os.getenv("FACE_ID_MIN_PERSON_CROP_HEIGHT", "64"))

#: The cooldown dict is keyed on (camera_id, local_track_id) and track ids
#: are never reused, so it would grow without bound over a long run. Capped
#: and pruned oldest-first - the same bounded-cache treatment
#: test_production_hardening_live.py applies to its own per-track cache.
FACE_ID_TRACK_CACHE_CAP = int(os.getenv("FACE_ID_TRACK_CACHE_CAP", "4096"))

#: tracking.py labels person tracks with this group name.
FACE_ID_PERSON_GROUP = os.getenv("FACE_ID_PERSON_GROUP", "person")

#: The dashboard feature key that arms Face-ID for a camera. multicam_inf.py
#: reads it out of /api/ai/cameras/ and calls set_enabled_cameras() - the
#: same path ppe_compliance and behaviour_analysis already take.
FACE_ID_CAMERA_FEATURE = "face_detection"


# ============================================================
# OBSERVATION
# ============================================================

class FaceObservation:
    """
    One immutable snapshot handed across the thread boundary.

    Immutable by construction, not by convention: `crop` is already a COPY
    made in submit() (see _crop_person), because frame[y1:y2, x1:x2] is a
    VIEW into a buffer cctv.py reuses for the next frame - a worker reading
    a view would see whatever pixels arrived after it was queued. Everything
    else here is a str/float. Nothing on this object is written after
    construction.
    """

    __slots__ = ("camera_id", "local_track_id", "crop", "timestamp",
                 "confidence", "submitted_at", "crop_origin", "person_bbox")

    def __init__(self, camera_id, local_track_id, crop, timestamp,
                 confidence=None, submitted_at=None, crop_origin=None,
                 person_bbox=None):
        self.camera_id = camera_id
        self.local_track_id = local_track_id
        self.crop = crop
        self.timestamp = timestamp
        self.confidence = confidence
        #: Top-left of `crop` within the source frame. A face box detected
        #: inside the crop is in CROP coordinates; the application stores
        #: FRAME coordinates, and this is the only place the offset is
        #: known - the frame itself is gone by the time the worker runs.
        self.crop_origin = crop_origin or (0, 0)
        #: The clamped person box this crop came from, in frame coordinates.
        self.person_bbox = person_bbox

    def queue_latency_ms(self):
        """How long this waited between submit() and the worker picking it
        up - the direct read on whether the worker is keeping up."""
        return (time.monotonic() - self.submitted_at) * 1000.0

    def __repr__(self):
        h, w = (self.crop.shape[:2] if self.crop is not None else (0, 0))
        return (f"<FaceObservation {self.camera_id}/{self.local_track_id} "
                f"crop={w}x{h}>")


# ============================================================
# STATS
# ============================================================

class _Stats:
    """
    Counters this ADAPTER owns - queue and lifecycle only.

    Deliberately does NOT include skipped_quality / embedding_errors /
    matches / new_identities / uncertain / qdrant_*. Those describe
    decisions this class cannot see and must not pretend to: they belong to
    the processor, and stats() below merges the processor's own counters in
    rather than this class maintaining a second, drifting copy of them.
    """

    __slots__ = ("submitted", "dropped_queue_full", "skipped_cooldown",
                 "skipped_ineligible", "processed", "worker_errors",
                 "heartbeats")

    def __init__(self):
        self.heartbeats = 0
        self.submitted = 0
        self.dropped_queue_full = 0
        self.skipped_cooldown = 0
        self.skipped_ineligible = 0
        self.processed = 0
        self.worker_errors = 0

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


class _HeartbeatTick:
    """A queue item that carries no crop: "this camera is still live".

    Goes through the same queue as an observation so the network call it
    leads to happens on the worker thread, never on the inference thread,
    and so a saturated queue drops it exactly like an observation.
    """

    __slots__ = ("camera_id", "timestamp")

    def __init__(self, camera_id, timestamp):
        self.camera_id = camera_id
        self.timestamp = timestamp


class _NullProcessor:
    """
    Phase 1's processor: proves the pipe works end to end without loading a
    single model. Phase 2 replaces this with face_id_manager.FaceIDManager,
    which has the same one-method surface.
    """

    def __init__(self):
        self.seen = 0

    def process(self, observation):
        self.seen += 1

    def stats(self):
        return {"null_processor_seen": self.seen}


# ============================================================
# PRODUCTION FACTORY
# ============================================================

def build_production_adapter():
    """
    The single call multicam_inf.py makes at startup.

    Returns an adapter that is ARMED BY THE DASHBOARD, not by this call.
    Nothing expensive happens here: no InsightFace, no Qdrant client, no HTTP
    session, no thread. Those are built on FIRST ARM, when
    set_enabled_cameras() is first given a non-empty set - the same lazy
    shape ppe_adapter.PPEAdapter._ensure_model() uses, and for the same
    reason: a config refresh must never block for seconds while a model
    loads, and a pipeline whose cameras all have Face Detection OFF must
    never load one at all.

    FACE_ID_ENABLED is an emergency kill switch, not the arming mechanism.
    When it is off this says so once, at startup, so "the dashboard says ON
    but nothing happens" is never a silent state.
    """
    if not FACE_ID_ENABLED:
        print("[FACE-ID] DISABLED BY FACE_ID_ENABLED=0 - the dashboard's "
              "per-camera Face Detection toggles are being IGNORED for this "
              "process. Unset the variable to return control to the dashboard.")
        return FaceIDAdapter(enabled=False)

    def factory():
        """Built once, on first arm. Imports live here so a pipeline that
        never arms a camera never imports them."""
        import face_id_manager
        import face_id_sink

        sink = face_id_sink.DashboardFaceSink()
        manager = face_id_manager.FaceIDManager(result_sink=sink.record)
        manager._sink = sink          # so close() can reach it
        return manager

    return FaceIDAdapter(processor_factory=factory, enabled=True)


# ============================================================
# ADAPTER
# ============================================================

class FaceIDAdapter:
    """
    The boundary. multicam_inf.py (Phase 6) and the PoC harness only ever
    call submit() / close() / stats().

    Constructed inert when disabled: no thread, no queue consumer, and
    submit()/close() become no-ops that still never raise. That mirrors
    reid_adapter.py's own "no camera authorised -> nothing initialised"
    behaviour, so a disabled Face-ID costs one boolean test per call.
    """

    def __init__(self, processor=None, enabled=None, enabled_cameras=None,
                 queue_size=None, cooldown_seconds=None,
                 shutdown_flush_seconds=None, stats_interval_seconds=None,
                 processor_factory=None):
        self.enabled = FACE_ID_ENABLED if enabled is None else bool(enabled)

        #: Cameras the DASHBOARD has armed, via set_enabled_cameras(). None
        #: means "every camera the caller submits", which only tests and the
        #: standalone harness use; production always passes an explicit set,
        #: so an unlisted camera is never observed.
        self._enabled_cameras = (
            None if enabled_cameras is None else frozenset(c.upper() for c in enabled_cameras)
        )
        #: Guards _enabled_cameras and the lazy processor build, both of which
        #: are written by the config-refresh thread and read by the inference
        #: thread on every submit().
        self._camera_lock = threading.Lock()
        self._processor_factory = processor_factory
        self._processor_failed = False

        self.queue_size = FACE_ID_QUEUE_SIZE if queue_size is None else queue_size
        self.cooldown_seconds = (
            FACE_ID_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
        )
        self.shutdown_flush_seconds = (
            FACE_ID_SHUTDOWN_FLUSH_SECONDS if shutdown_flush_seconds is None
            else shutdown_flush_seconds
        )
        self.stats_interval_seconds = (
            FACE_ID_STATS_INTERVAL_SECONDS if stats_interval_seconds is None
            else stats_interval_seconds
        )

        self._stats = _Stats()
        self._last_submit = {}
        #: camera_id -> monotonic time of its last queued heartbeat.
        self._last_heartbeat = {}
        self._queue = None
        self._thread = None
        self._stop_event = threading.Event()
        self._closed = False
        self._last_stats_print = time.monotonic()
        #: With a factory, the processor stays None until a camera is armed.
        #: Without one, behaviour is unchanged from before: an injected
        #: processor, or the null processor for Phase-1 style use.
        if processor is not None:
            self._processor = processor
        elif processor_factory is not None:
            self._processor = None
        else:
            self._processor = _NullProcessor()

        if not self.enabled:
            print("[FACE-ID] disabled - no queue, no worker, no model")
            return

        self._queue = queue.Queue(maxsize=self.queue_size)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="face-id-worker",
        )
        self._thread.start()

        if self._processor_factory is not None:
            # Production: nothing is armed until the dashboard says so, and
            # saying "cameras=all" here would misdescribe that.
            print(f"[FACE-ID] ready  queue={self.queue_size}  "
                  f"cooldown={self.cooldown_seconds}s  "
                  f"awaiting per-camera Face Detection configuration from the dashboard")
        else:
            cameras = ("all" if self._enabled_cameras is None
                       else ",".join(sorted(self._enabled_cameras)) or "none")
            print(f"[FACE-ID] enabled  queue={self.queue_size}  "
                  f"cooldown={self.cooldown_seconds}s  cameras={cameras}  "
                  f"processor={type(self._processor).__name__}")

    # ------------------------------------------------------------- arming
    def set_enabled_cameras(self, camera_ids):
        """
        Which cameras have Face Detection armed, from their dashboard feature
        flags. This is the ONLY thing that arms Face-ID for a camera.

        Called at startup and on every config refresh, so switching Face
        Detection on or off in the dashboard takes effect within one refresh
        interval without restarting inference - the same contract
        ppe_adapter.PPEAdapter.set_enabled_cameras() offers, deliberately, so
        an operator learns one behaviour rather than two.

        The model is built HERE on first arm, never in the inference loop and
        never at startup, and outside the lock: loading InsightFace takes
        seconds and nothing may hold a config refresh for that long.
        """
        wanted = frozenset(c.upper() for c in (camera_ids or ()))

        with self._camera_lock:
            changed = wanted != (self._enabled_cameras or frozenset())
            self._enabled_cameras = wanted
            needs_processor = bool(wanted) and self._processor is None

        if changed:
            print(f"[FACE-ID] camera configuration: "
                  f"{', '.join(f'{c}=ON' for c in sorted(wanted)) or 'no camera armed'}")

        if needs_processor and self._processor_factory is not None and not self._processor_failed:
            try:
                processor = self._processor_factory()
            except Exception as exc:  # noqa: BLE001 - a config refresh must never raise
                # Latched, so a broken model is not retried on every refresh -
                # that would turn one failure into a repeating stall.
                self._processor_failed = True
                print(f"[FACE-ID] could not start Face-ID ({type(exc).__name__}: {exc}) "
                      f"- Face-ID stays off, CCTV inference is unaffected")
                return

            with self._camera_lock:
                if self._processor is None:
                    self._processor = processor
            print(f"[FACE-ID] armed  processor={type(processor).__name__}  "
                  f"cameras={', '.join(sorted(wanted))}")

    def camera_armed(self, camera_id):
        with self._camera_lock:
            if self._enabled_cameras is None:
                return True
            # ZAYED: set_enabled_cameras() stores UPPER-CASE ids; compare the same way (as the
            # fall and fight adapters do). Dubai's "CAM-R25" ids were already upper case, so
            # the lower-case Zayed "camera_01" exposed this: Face-ID observed nothing.
            return bool(camera_id) and camera_id.upper() in self._enabled_cameras

    # ------------------------------------------------------------- producer
    def submit(self, camera_id, tracks, frame, timestamp):
        """
        Call once per camera, per tick, right after track_manager.update().

        Cheap and non-blocking by construction: a cooldown lookup, a numpy
        slice-and-copy, and a put_nowait() per eligible track. Runs NO
        model, opens NO socket, and never waits on the worker. Always safe
        to call unconditionally, enabled or not.

        Returns the number of observations actually enqueued this call -
        useful to a harness, ignorable by production (reid_adapter.flush()
        sets the same precedent of a return value Phase 1 callers need not
        read).
        """
        if not self.enabled or self._queue is None:
            return 0

        # The disarmed path, and the one that must stay cheap: a set lookup
        # under an uncontended lock, then return. No crop, no copy, no queue
        # touch, and - because the processor is built lazily - no model has
        # been loaded for this process at all if nothing was ever armed.
        with self._camera_lock:
            armed = self._enabled_cameras is None or (
                bool(camera_id) and camera_id.upper() in self._enabled_cameras)   # ZAYED: see camera_armed()
            has_processor = self._processor is not None

        if not armed or not has_processor:
            return 0

        queued = 0
        now = time.monotonic()

        # Before the tracks, and independent of them: an empty view is
        # exactly when this matters.
        self._maybe_queue_heartbeat(camera_id, timestamp, now)

        # Per-track try/except, NOT one guard around the whole loop: a
        # single malformed track (missing attribute, odd bbox type) must not
        # shadow every OTHER track on this camera this tick. Same reasoning
        # as reid_adapter.process_tracks().
        for tracked in tracks:
            try:
                if getattr(tracked, "group", None) != FACE_ID_PERSON_GROUP:
                    continue

                track_id = tracked.track_id
                key = (camera_id, track_id)

                last = self._last_submit.get(key)
                if last is not None and (now - last) < self.cooldown_seconds:
                    self._stats.skipped_cooldown += 1
                    continue

                cropped = self._crop_person(frame, tracked.bbox)
                if cropped is None:
                    self._stats.skipped_ineligible += 1
                    continue

                crop, box = cropped

                observation = FaceObservation(
                    camera_id=camera_id,
                    local_track_id=track_id,
                    crop=crop,
                    timestamp=timestamp,
                    confidence=getattr(tracked, "confidence", None),
                    crop_origin=(box[0], box[1]),
                    person_bbox=box,
                )

                try:
                    self._queue.put_nowait(observation)
                except queue.Full:
                    # Drop, count, carry on. Never block the inference
                    # thread to guarantee delivery: a dropped observation
                    # costs one sample of a face that will very likely be
                    # seen again next cooldown, whereas blocking here would
                    # reintroduce exactly the coupling this class removes.
                    self._stats.dropped_queue_full += 1
                    continue

                # Only marked AFTER a successful enqueue, so an observation
                # lost to a full queue does not also silence this track for
                # a whole cooldown period.
                self._last_submit[key] = now
                self._stats.submitted += 1
                queued += 1

            except Exception as exc:  # noqa: BLE001 - never reaches the caller
                self._stats.skipped_ineligible += 1
                self._log_error("submit", exc)

        self._prune_track_cache()
        self._maybe_print_stats()

        return queued

    def _maybe_queue_heartbeat(self, camera_id, timestamp, now):
        """Queue at most one heartbeat per camera per interval.

        Cheap by construction - a dict lookup and, at most once per interval,
        a put_nowait of an object holding two scalars. A full queue drops it
        silently: the next tick will do, and an observation must never lose
        its place to a heartbeat.
        """
        last = self._last_heartbeat.get(camera_id)
        if last is not None and (now - last) < FACE_ID_HEARTBEAT_SECONDS:
            return

        try:
            self._queue.put_nowait(_HeartbeatTick(camera_id, timestamp))
        except queue.Full:
            self._stats.dropped_queue_full += 1
            return
        except Exception as exc:  # noqa: BLE001 - never reaches the caller
            self._log_error("heartbeat", exc)
            return

        self._last_heartbeat[camera_id] = now
        self._stats.heartbeats += 1

    def clear_cooldown(self, camera_id, local_track_id):
        """
        Forget one track's cooldown so its next submit() is accepted
        immediately.

        The seam Phase 2 needs: an UNCERTAIN decision resolved nothing, so
        that track should get another attempt on the next frame rather than
        sitting unresolved for a full cooldown. The prototype
        (test_face.py) established this behaviour by never caching an
        UNCERTAIN result; expressed here as an explicit call so the worker
        can ask for a retry without this class knowing what UNCERTAIN means.
        """
        self._last_submit.pop((camera_id, local_track_id), None)

    # -------------------------------------------------------------- worker
    def _run(self):
        """
        The one worker thread. Deliberately one, not a pool: a full
        InsightFace pass was measured at ~6.4 ms median / ~7.9 ms mean on
        this hardware, so a single thread sustains ~125 observations/second
        - far above what per-track cooldown can generate. A pool would add
        GPU contention and ordering hazards to buy throughput nothing needs.
        """
        while True:
            try:
                observation = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue

            self._process_one(observation)

    def _process_one(self, observation):
        """
        Per-OBSERVATION exception containment, which is the part that
        differs from AsyncUpsertStore's per-item retry loop.

        A face pass is a much larger operation than a Qdrant write - a
        detector, an embedder, two attribute heads and a network search -
        so there are many more ways for ONE crop to fail (a degenerate
        bbox, a model quirk, a transient CUDA error) and no reason to
        believe an immediate retry of that same crop would do better. The
        item is therefore counted and dropped, never retried here, and the
        worker moves on. One bad observation must not end Face-ID for the
        rest of the process's life.
        """
        try:
            processor = self._processor
            if processor is None:
                return              # disarmed between enqueue and dequeue

            if isinstance(observation, _HeartbeatTick):
                # A processor without heartbeat() (Phase 1's _NullProcessor, a
                # harness stub) simply has nothing to report, and must not
                # raise on its way past.
                report = getattr(processor, "heartbeat", None)
                if report is not None:
                    report(observation.camera_id, observation.timestamp)
                return

            processor.process(observation)
            self._stats.processed += 1
        except Exception as exc:  # noqa: BLE001 - must never kill this thread
            self._stats.worker_errors += 1
            self._log_error("worker", exc)

    # ------------------------------------------------------------ lifecycle
    def close(self):
        """
        Bounded, idempotent shutdown.

        Signals the worker to exit once the queue is drained, then waits at
        most shutdown_flush_seconds for it. NEVER waits indefinitely: the
        worker may be inside a face model or a Qdrant call, and the primary
        pipeline's shutdown must not hang on Face-ID. The thread is a
        daemon, so anything still running at the timeout does not hold the
        process open either.

        Every step is guarded, including against KeyboardInterrupt, which
        does NOT inherit from Exception - a SIGINT landing inside the join()
        below would otherwise escape close() entirely and skip the caller's
        remaining shutdown steps. That failure was found and fixed for real
        in the Re-ID adapter's own close(); this inherits the lesson rather
        than rediscovering it.
        """
        if self._closed or not self.enabled:
            self._closed = True
            return

        self._closed = True
        self._stop_event.set()

        try:
            if self._thread is not None:
                self._thread.join(timeout=self.shutdown_flush_seconds)
        except KeyboardInterrupt:
            print("[FACE-ID] interrupted while waiting for the worker - continuing shutdown")
        except Exception as exc:  # noqa: BLE001
            self._log_error("close.join", exc)

        try:
            remaining = self._queue.qsize() if self._queue is not None else 0
            stats = self.stats()
            print(f"[FACE-ID] shutdown  submitted={stats['submitted']} "
                  f"processed={stats['processed']} "
                  f"dropped_queue_full={stats['dropped_queue_full']} "
                  f"skipped_cooldown={stats['skipped_cooldown']} "
                  f"skipped_ineligible={stats['skipped_ineligible']} "
                  f"worker_errors={stats['worker_errors']} "
                  f"unprocessed_at_exit={remaining}")
        except Exception as exc:  # noqa: BLE001
            self._log_error("close.report", exc)

        try:
            closer = getattr(self._processor, "close", None) if self._processor else None
            if callable(closer):
                closer()
        except KeyboardInterrupt:
            print("[FACE-ID] interrupted while closing the processor - continuing shutdown")
        except Exception as exc:  # noqa: BLE001
            self._log_error("close.processor", exc)

    # ------------------------------------------------------------- reporting
    def queue_depth(self):
        return self._queue.qsize() if self._queue is not None else 0

    def stats(self):
        """Adapter counters, plus the processor's own if it exposes any.

        Merged rather than duplicated: the processor owns the decision
        counters (skipped_quality, matches, new_identities, uncertain,
        qdrant_*) because it is the only thing that can observe them.
        """
        merged = self._stats.as_dict()
        merged["queue_depth"] = self.queue_depth()

        try:
            processor_stats = getattr(self._processor, "stats", None) if self._processor else None
            if callable(processor_stats):
                merged.update(processor_stats())
        except Exception as exc:  # noqa: BLE001 - reporting must never raise
            self._log_error("stats", exc)

        return merged

    def _maybe_print_stats(self):
        if self.stats_interval_seconds <= 0:
            return

        now = time.monotonic()
        if (now - self._last_stats_print) < self.stats_interval_seconds:
            return

        self._last_stats_print = now
        stats = self.stats()
        print(f"[FACE-ID-QUEUE] depth={stats['queue_depth']}/{self.queue_size} "
              f"submitted={stats['submitted']} processed={stats['processed']} "
              f"dropped_queue_full={stats['dropped_queue_full']} "
              f"skipped_cooldown={stats['skipped_cooldown']} "
              f"worker_errors={stats['worker_errors']}")

    @staticmethod
    def _log_error(where, exc):
        print(f"[FACE-ID] {where} error (contained): {type(exc).__name__}: {exc}")

    # --------------------------------------------------------------- crops
    @staticmethod
    def _crop_person(frame, bbox):
        """
        Clamp the bbox to the frame and return (COPY, clamped_box), or None.

        The clamped box travels with the crop because the worker needs frame
        coordinates to record where a face was, and by the time it runs the
        frame is gone.

        The .copy() is the load-bearing line in this method. frame[y1:y2,
        x1:x2] is a numpy VIEW sharing memory with the frame buffer, and
        cctv.py reuses those buffers for subsequent frames - so a worker
        reading a view could embed pixels from a completely different
        moment, and the resulting identity decision would be silently wrong
        rather than loudly broken. Copying at the boundary is what makes
        FaceObservation genuinely immutable.

        Deliberately NOT reid_poc/reid.crop_person(): that applies
        body-Re-ID padding and an aspect-ratio validity rule tuned for OSNet
        full-body crops, and pulls reid_poc/config.py into this module's
        import graph. The simple clamp here matches what the test_face.py
        prototype validated for the face path.
        """
        if frame is None or bbox is None:
            return None

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))

        if x2 <= x1 or y2 <= y1:
            return None

        if (x2 - x1) < FACE_ID_MIN_PERSON_CROP_WIDTH:
            return None
        if (y2 - y1) < FACE_ID_MIN_PERSON_CROP_HEIGHT:
            return None

        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _prune_track_cache(self):
        """Keep the cooldown dict bounded. Track ids are never reused, so
        without this it grows for the life of the process; oldest submit
        times go first, which are exactly the tracks least likely to still
        be in frame."""
        if len(self._last_submit) <= FACE_ID_TRACK_CACHE_CAP:
            return

        excess = len(self._last_submit) - FACE_ID_TRACK_CACHE_CAP
        for key in sorted(self._last_submit, key=self._last_submit.get)[:excess]:
            del self._last_submit[key]
