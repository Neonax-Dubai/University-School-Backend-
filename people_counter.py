"""
People counting - how many people each armed camera sees, published to the
dashboard as short periodic samples.

    counter = PeopleCounter()
    counter.set_enabled_cameras({"CAM-R02": [zone, ...]})       # every refresh
    counter.observe(camera_id, tracked_objects, width, height)  # every frame

    publisher = PeopleCountPublisher(counter, dashboard_url, token)
    publisher.start()      # one daemon thread
    ...
    publisher.stop()

WHAT IS COUNTED
---------------
Confirmed PERSON tracks matched in the frame - the same objects crowd detection
counts - never raw YOLO boxes, so a person the detector boxes twice is still
one person. No model runs here: it reads the tracks the loop already produced.

TrackManager.update() returns only the tracks MATCHED in this frame, so one
missed detection makes that frame's count dip by one. A sample therefore
reports the median of the per-frame counts across its window as the headline
figure, which a single missed frame cannot move, with min / max / mean beside
it. median_high rather than a plain median: the count is an integer, and the
error it absorbs (missed detections) only ever pulls the count down.

unique_tracks is the number of distinct track ids seen in the window. It is an
upper bound on different people, not a headcount: a track that fragments
behind an occlusion is counted twice. new_tracks is how many of those were
first confirmed inside the window, with the same caveat.

Zones are counted with the anchor rule crowd detection uses (zones.ANCHOR), so
a zone's people count and its crowd alarm can never disagree about who is in it.

WHY SAMPLES AND NOT EVENTS
--------------------------
A count is state, not an incident. Posting it as events would put 8,640 rows a
day per camera into the event table and the alarm path. It follows the camera
health telemetry pattern instead: a timer, no queue and no retry. A POST that
fails loses that 10 s window, and the chart shows a gap rather than old data
presented as new.

A window in which the camera produced no processed frame reports frames=0 and
null counts. No video must never read as "nobody there".

WHY IT CANNOT AFFECT INFERENCE
------------------------------
observe() is a dict lookup for an unarmed camera and a few comparisons for an
armed one, and its whole body is guarded. The network I/O happens on the
publisher's own daemon thread with a bounded timeout, and every exception is
caught inside that loop.
"""
import os
import statistics
import threading
import time

import requests

import tracking
import zones as zone_geometry


#: The dashboard feature flag that arms counting for a camera.
PEOPLE_COUNTING_FEATURE = "people_counting"

#: Seconds per sample. The dashboard treats a camera whose newest sample is
#: older than PEOPLE_COUNT_FRESH_SECONDS (45 s) as stale, so a single failed
#: publish cannot make a live camera read as stale.
PUBLISH_INTERVAL_SECONDS = float(os.getenv("PEOPLE_COUNT_PUBLISH_SECONDS", "10.0"))

#: Bounded, and shorter than the interval: a hung dashboard must never leave
#: this thread holding on past the next cycle.
POST_TIMEOUT = float(os.getenv("PEOPLE_COUNT_POST_TIMEOUT", "5.0"))

#: Only the first of a run of identical failures is logged, plus a reminder.
FAILURE_LOG_REPEAT_SECONDS = 300.0


class _Window:
    """Everything observed for one camera since its last sample."""

    __slots__ = ("started_at", "counts", "track_ids", "new_tracks", "zone_counts")

    def __init__(self, started_at):
        self.started_at = started_at
        self.counts = []            # per-frame person count
        self.track_ids = set()      # distinct person track ids
        self.new_tracks = 0         # tracks first confirmed in this window
        self.zone_counts = {}       # zone_id -> [per-frame count]


def _summarise(values):
    """(median, min, max, mean) of per-frame counts, or all None for no frames."""
    if not values:
        return None, None, None, None
    return (
        statistics.median_high(values),
        min(values),
        max(values),
        round(sum(values) / len(values), 2),
    )


class PeopleCounter:
    """Per-camera windows of person counts, drained into samples."""

    def __init__(self, clock=time.time, log=print):
        self._clock = clock
        self._log = log
        self._lock = threading.Lock()

        # camera_id -> [zone dict]. Replaced wholesale, never mutated, so the
        # membership test in observe() needs no lock.
        self._zones = {}

        # camera_id -> _Window
        self._windows = {}

        self.frames_observed = 0
        self.errors = 0
        self._error_logged = False

    # --------------------------------------------------------------- config

    def set_enabled_cameras(self, camera_zones):
        """Arm exactly these cameras. camera_zones: {camera_id: [zone dict]}.

        Only enabled zones with a real polygon are counted. A camera that is
        disarmed loses its open window; one that stays armed keeps it, so a
        refresh that changes nothing does not cut a sample short.
        """
        now = self._clock()
        armed = {
            camera_id: [
                zone for zone in (camera_zone_list or [])
                if zone.get("enabled", True)
                and len(zone.get("coordinates") or []) >= 3
            ]
            for camera_id, camera_zone_list in camera_zones.items()
        }

        with self._lock:
            self._zones = armed
            for camera_id in [c for c in self._windows if c not in armed]:
                del self._windows[camera_id]
            for camera_id in armed:
                if camera_id not in self._windows:
                    self._windows[camera_id] = _Window(now)

    def enabled_cameras(self):
        return sorted(self._zones)

    def camera_armed(self, camera_id):
        return camera_id in self._zones

    # -------------------------------------------------------------- observe

    def observe(self, camera_id, tracked_objects, frame_width, frame_height):
        """Count one processed frame. Never raises."""
        zone_list = self._zones.get(camera_id)
        if zone_list is None:
            return

        try:
            persons = {
                tracked.track_id: tracked
                for tracked in tracked_objects
                if tracked.group == tracking.PERSON_GROUP
            }
            new_tracks = sum(1 for tracked in persons.values() if tracked.is_new)

            zone_counts = {}
            for zone in zone_list:
                polygon = zone.get("coordinates") or []
                inside = 0
                for tracked in persons.values():
                    point = zone_geometry.anchor_point(
                        tracked.bbox, frame_width, frame_height)
                    if point is not None and zone_geometry.point_in_polygon(
                            point[0], point[1], polygon):
                        inside += 1
                zone_counts[zone.get("id")] = inside

            with self._lock:
                window = self._windows.get(camera_id)
                if window is None:          # disarmed between lookup and lock
                    return
                window.counts.append(len(persons))
                window.track_ids.update(persons)
                window.new_tracks += new_tracks
                for zone_id, inside in zone_counts.items():
                    window.zone_counts.setdefault(zone_id, []).append(inside)
                self.frames_observed += 1

        except Exception as exc:                                 # noqa: BLE001
            self.errors += 1
            if not self._error_logged:
                self._error_logged = True
                self._log(f"[PEOPLE_COUNT] observe error (logged once): "
                          f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- drain

    def drain(self):
        """Close every armed camera's window; one sample dict per camera."""
        now = self._clock()
        samples = []

        with self._lock:
            for camera_id, zone_list in self._zones.items():
                window = self._windows.get(camera_id) or _Window(now)
                self._windows[camera_id] = _Window(now)

                count, count_min, count_max, count_avg = _summarise(window.counts)

                zone_samples = []
                for zone in zone_list:
                    zone_count, _, zone_max, _ = _summarise(
                        window.zone_counts.get(zone.get("id"), []))
                    zone_samples.append({
                        "zone_id": zone.get("id"),
                        "name": zone.get("name"),
                        "count": zone_count,
                        "count_max": zone_max,
                    })

                samples.append({
                    "camera_id": camera_id,
                    "window_start": round(window.started_at, 3),
                    "measured_at": round(now, 3),
                    "frames": len(window.counts),
                    "count": count,
                    "count_min": count_min,
                    "count_max": count_max,
                    "count_avg": count_avg,
                    "unique_tracks": len(window.track_ids),
                    "new_tracks": window.new_tracks,
                    "zones": zone_samples,
                })

        return samples


class PeopleCountPublisher:
    """Periodically POSTs drained samples to the dashboard."""

    def __init__(self, counter, dashboard_url, token,
                 interval=PUBLISH_INTERVAL_SECONDS, timeout=POST_TIMEOUT,
                 session=None, log=print):
        self.counter = counter
        self.url = f"{dashboard_url.rstrip('/')}/api/ai/people-counts/"
        self.token = token
        self.interval = float(interval)
        self.timeout = float(timeout)
        self._session = session
        self._log = log

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.cycles = 0
        self.sent = 0               # successful POSTs
        self.samples_sent = 0       # samples accepted by the dashboard
        self.failed = 0             # failed POSTs
        self.skipped_empty = 0      # no camera armed
        self.errors = 0             # unexpected faults inside the loop
        self._last_failure = None
        self._last_failure_at = 0.0

    # ------------------------------------------------------------ lifecycle

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="people-count-publisher", daemon=True)
        self._thread.start()
        self._log(f"[PEOPLE_COUNT] publisher -> {self.url} "
                  f"every {self.interval:.0f}s")

    def stop(self, timeout=5.0):
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    # --------------------------------------------------------------- worker

    def _run(self):
        while not self._stop.is_set():
            # Wait FIRST, so the first sample covers a full window.
            if self._stop.wait(self.interval):
                return
            try:
                self.publish_once()
            except Exception as exc:                  # never kills the thread
                with self._lock:
                    self.errors += 1
                self._log_failure(f"publisher fault: {type(exc).__name__}: {exc}")

    def publish_once(self):
        """One cycle. Returns True when the samples were accepted."""
        with self._lock:
            self.cycles += 1
        samples = self.counter.drain()
        if not samples:
            with self._lock:
                self.skipped_empty += 1
            return False
        return self._post({"samples": samples})

    def _post(self, payload):
        session = self._session
        if session is None:
            session = self._session = requests.Session()
            session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Token {self.token}",
            })
        try:
            response = session.post(self.url, json=payload, timeout=self.timeout)
        except Exception as exc:
            with self._lock:
                self.failed += 1
            self._log_failure(f"{type(exc).__name__}: {exc}")
            return False

        if response.status_code in (200, 201, 202):
            with self._lock:
                self.sent += 1
                self.samples_sent += len(payload["samples"])
                self._last_failure = None
            return True

        with self._lock:
            self.failed += 1
        body = ""
        try:
            body = response.text[:200].replace("\n", " ")
        except Exception:                                        # noqa: BLE001
            pass
        self._log_failure(f"HTTP {response.status_code} {body}")
        return False

    def _log_failure(self, detail):
        now = time.time()
        with self._lock:
            same = (detail == self._last_failure)
            recent = (now - self._last_failure_at) < FAILURE_LOG_REPEAT_SECONDS
            if same and recent:
                return
            self._last_failure = detail
            self._last_failure_at = now
        self._log(f"[PEOPLE_COUNT] publish failed: {detail}")

    # ---------------------------------------------------------------- stats

    def status_line(self):
        with self._lock:
            return (f"PeopleCount: cameras={len(self.counter.enabled_cameras())} "
                    f"frames={self.counter.frames_observed} "
                    f"cycles={self.cycles} sent={self.sent} "
                    f"samples={self.samples_sent} failed={self.failed} "
                    f"errors={self.errors + self.counter.errors} "
                    f"every={self.interval:.0f}s")
