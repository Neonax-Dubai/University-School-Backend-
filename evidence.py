"""
Evidence crops - the visual proof attached to an AI event.

Policy (deliberately narrow):
    * ONE crop per newly confirmed track (P-xxxx / V-xxxx), on its first event.
    * ONE crop per alarm-class event (intrusion, perimeter_breach, loitering,
      line_crossing, crowd_detected, camera_*).
Not one per frame, not one per debounced person_detected. At full estate that
distinction is the difference between ~35 GB/day and ~420 GB/day.

Where the work happens
----------------------
The CROP is taken on the inference thread, because that is the only place the
frame exists - Django cannot re-pull it later, the moment is gone. It is cheap:
a NumPy slice plus a JPEG encode of a person-sized patch, well under a
millisecond, and it is bounded by the new-track rate, not the frame rate.

The UPLOAD is not. It goes on a bounded queue drained by worker threads, and
only the encoded bytes are queued (~15 KB), never the 6 MB frame.

Ordering
--------
The crop is uploaded BEFORE its event is POSTed, so the dashboard never holds a
row pointing at an object that does not exist. The evidence worker owns that
sequence and then hands the event to the normal event sender. If the upload
fails the event is still sent, just without an evidence reference - a storage
outage must never cost us the detection.

Env overrides:
  EVIDENCE_ENABLED=0        stop generating crops entirely
  EVIDENCE_BACKEND          filer | local        (default filer)
  SEAWEED_FILER_URL         default http://127.0.0.1:8888
  EVIDENCE_LOCAL_DIR        used when backend=local (validation/offline)
  EVIDENCE_PADDING          fraction of bbox added on each side (default 0.12)
  EVIDENCE_JPEG_QUALITY     default 85
  EVIDENCE_MIN_PIXELS       skip crops smaller than this on either side (24)
  EVIDENCE_TTL              SeaweedFS TTL applied at upload (default 7d)
  EVIDENCE_WORKERS          upload threads (default 2)
  EVIDENCE_QUEUE_SIZE       pending uploads held in memory (default 500)
"""

import os
import threading
import time
import uuid
from datetime import datetime
from queue import Queue, Empty, Full

import cv2
import requests


# ============================================================
# CONFIGURATION
# ============================================================

ENABLED = os.getenv("EVIDENCE_ENABLED", "1") == "1"
BACKEND = os.getenv("EVIDENCE_BACKEND", "filer").lower()

# The master's /dir/assign hands back Docker-internal hostnames
# (seaweed-volume:8080) that do not resolve on this host, so the filer is the
# usable interface. It is bound to 127.0.0.1 - browsers never touch it, Django
# proxies it behind authentication.
FILER_URL = os.getenv("SEAWEED_FILER_URL", "http://127.0.0.1:8888")

LOCAL_DIR = os.getenv("EVIDENCE_LOCAL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_out"))

PADDING = float(os.getenv("EVIDENCE_PADDING", "0.12"))
JPEG_QUALITY = int(os.getenv("EVIDENCE_JPEG_QUALITY", "85"))
MIN_PIXELS = int(os.getenv("EVIDENCE_MIN_PIXELS", "24"))

# Retention lever. SeaweedFS expires the object itself - no cleanup cron.
# PoC target is 7 days; production retention per data class belongs here.
TTL = os.getenv("EVIDENCE_TTL", "30d")
# ZAYED: evidence lives under its own prefix in SeaweedFS, e.g.
# zayed-evidence/2026/09/22/camera_01/P-2K7Q-0017/<uid>/crop.jpg
KEY_PREFIX = os.getenv("EVIDENCE_KEY_PREFIX", "zayed-evidence").strip("/")

WORKERS = int(os.getenv("EVIDENCE_WORKERS", "2"))
QUEUE_SIZE = int(os.getenv("EVIDENCE_QUEUE_SIZE", "500"))

UPLOAD_TIMEOUT = float(os.getenv("EVIDENCE_UPLOAD_TIMEOUT", "5.0"))

# How many times a single track may retry its representative crop before the
# pipeline gives up and reports it honestly as uncaptured. A track that is
# always too small or always at the frame edge would otherwise re-attempt on
# every event for its whole life.
MAX_CAPTURE_ATTEMPTS = int(os.getenv("EVIDENCE_MAX_ATTEMPTS", "12"))

FAILURE_LOG_INTERVAL = 15.0


# Event types that always warrant a crop, on top of first-sighting of a track.
ALARM_EVIDENCE_TYPES = {
    "intrusion",
    "perimeter_breach",
    "loitering",
    "line_crossing",
    "crowd_detected",
    "abandoned_object",
    # Not an alarm - no Alarm row is raised for it - but a vehicle-zone
    # detection exists to hand a picture of the vehicle to a future ANPR
    # stage, so an image is the whole point rather than a nicety. The crop is
    # taken from the VEHICLE track's own bbox in EventPipeline._emit.
    "vehicle_zone_detection",
    # Compliance findings, and neither raises an Alarm - but with several
    # people in one zone the video alone cannot say WHICH of them was the
    # subject. The crop is taken from the violating PERSON track's own bbox in
    # EventPipeline._emit, so each event carries a picture of its own subject
    # while the shared clip carries the temporal context.
    "ppe_violation",
    "behaviour_violation",
    # Not an alarm either, and no clip - but a still is the ONLY way an
    # operator can check the claim. "Police Uniform Detected, 53%" with no
    # picture is unverifiable: the class is provisional, and the whole point
    # of showing it is that a human can look and agree or disagree. The crop
    # is the PERSON's box (the same convention ppe_violation uses above), not
    # the uniform sub-box, so the operator sees who is being described.
    "police_uniform_detected",
    # Not an alarm and no clip. An object log row without a picture says
    # only "YOLO thought it saw a cell phone"; the still is what makes it a
    # record. object_logger.py hands over the object's box grown for context,
    # so small objects survive MIN_PIXELS.
    "object_detected",
    # Pair event: the crop is the UNION of both people's boxes, taken in
    # EventPipeline.handle_camera_event, so the still shows the two of them
    # together - which is the whole evidence of a distancing violation.
    "physical_distancing_violation",
    # Alarms with no inference-time picture until now: their workers are
    # asynchronous, so the frame never reached handle_camera_event and the
    # only still was the clip poster ~25 s later (or never, if the clip
    # failed). The worker now attaches its decision frame to the finding.
    "fall_detected",
    "violence_detected",
    # ONE camera-health event type; the detector that noticed (defocus,
    # obstruction, signal loss) arrives as metadata["reason"].
    "camera_tamper",
}


# ============================================================
# CROP
# ============================================================

#: Set once if the overlay ever raises, so the warning is printed one time.
_OVERLAY_WARNED = False


def crop_bbox(frame, bbox, padding=PADDING, quality=JPEG_QUALITY, overlay=None):
    """
    Cut a padded bbox out of the frame and JPEG-encode it.

    Returns (jpeg_bytes, width, height) or None if the box is unusable.
    The padding gives the operator context - a tightly cropped torso is much
    harder to identify than one with a bit of the scene around it - and is
    clamped to the frame, so a detection at the edge yields a smaller crop
    rather than an out-of-bounds slice.

    overlay, when given, is (event_type, metadata, contributors): the boxes
    that made the event are drawn on the cut before it is encoded, so the
    still shows WHICH people or which object the event counted (see
    evidence_overlay). It is one pass on an array already in memory - the
    same single encode as before - and it can never fail the crop: on any
    error the unannotated cut is returned.
    """
    if frame is None or bbox is None or len(bbox) != 4:
        return None

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = (int(v) for v in bbox)

    # Guard against inverted or degenerate boxes before padding them.
    if x2 <= x1 or y2 <= y1:
        return None

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(width, x2 + pad_x)
    cy2 = min(height, y2 + pad_y)

    if (cx2 - cx1) < MIN_PIXELS or (cy2 - cy1) < MIN_PIXELS:
        return None

    patch = frame[cy1:cy2, cx1:cx2]

    if patch.size == 0:
        return None

    if overlay is not None:
        try:
            import evidence_overlay

            event_type, metadata, contributors = overlay
            # .copy() first: the slice above is a VIEW of the live frame, and
            # drawing on it would paint boxes into the frame the rest of the
            # pipeline is still using.
            annotated = patch.copy()
            if evidence_overlay.annotate(annotated, event_type, metadata, bbox,
                                         origin=(cx1, cy1), contributors=contributors):
                patch = annotated
        except Exception as exc:                          # noqa: BLE001
            # Once per process: a broken overlay must not turn into a line
            # per frame in the consensus log, and it costs nobody the crop.
            global _OVERLAY_WARNED
            if not _OVERLAY_WARNED:
                _OVERLAY_WARNED = True
                print(f"[EVIDENCE-OVERLAY] disabled for this run after: "
                      f"{type(exc).__name__}: {exc}")

    ok, buffer = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, quality])

    if not ok:
        return None

    return buffer.tobytes(), int(cx2 - cx1), int(cy2 - cy1)


def build_storage_key(camera_id, track_id, observed_at, evidence_uid, kind="crop"):
    """
    cctv-events/2026/08/19/CAM-R01/P-0017/<uid>/crop.jpg

    Date, camera, track and evidence type are all deterministic in the path.
    The event id is not - Django assigns it when the row is written, after this
    upload - so the Event <-> Evidence link is carried by the database row, and
    the uid here is what ties the two together.
    """
    stamp = datetime.fromtimestamp(observed_at)

    safe_track = (track_id or "untracked").replace("/", "_")

    return (
        f"{KEY_PREFIX}/{stamp:%Y/%m/%d}/{camera_id}/{safe_track}/"
        f"{evidence_uid}/{kind}.jpg"
    )


# ============================================================
# STORAGE BACKENDS
# ============================================================

class FilerStore:
    """SeaweedFS filer over plain HTTP. Bound to localhost, never public."""

    name = "seaweedfs-filer"

    def __init__(self, base_url=FILER_URL, ttl=TTL, timeout=UPLOAD_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.ttl = ttl
        self.timeout = timeout
        self._local = threading.local()

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def put(self, storage_key, data, content_type="image/jpeg"):
        url = f"{self.base_url}/{storage_key.lstrip('/')}"

        params = {"ttl": self.ttl} if self.ttl else None

        response = self._session().post(
            url,
            params=params,
            files={"file": (os.path.basename(storage_key), data, content_type)},
            timeout=self.timeout,
        )

        if response.status_code not in (200, 201, 204):
            raise RuntimeError(f"filer HTTP {response.status_code}: {response.text[:160]}")

        return storage_key


class LocalStore:
    """Filesystem mirror of the same key layout - for validating the pipeline."""

    name = "local"

    def __init__(self, root=LOCAL_DIR):
        self.root = root

    def put(self, storage_key, data, content_type="image/jpeg"):
        path = os.path.join(self.root, storage_key)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as handle:
            handle.write(data)

        return storage_key


def make_store(backend=BACKEND):
    return LocalStore() if backend == "local" else FilerStore()


# ============================================================
# PIPELINE
# ============================================================

class EvidencePipeline:
    """
    Uploads a crop, attaches its reference to the event, then releases the
    event to the sender. Owns its own threads; the inference loop only calls
    submit().
    """

    def __init__(self, event_sender, store=None, workers=WORKERS, queue_size=QUEUE_SIZE):
        self.event_sender = event_sender
        self.store = store if store is not None else make_store()

        self._queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.uploaded = 0
        self.failed = 0
        self.dropped = 0
        self.skipped = 0
        self.exhausted = 0

        # (camera_id, track_id) -> attempts made, for tracks still without a
        # usable crop. A track disappears from here once one is submitted.
        #
        # This exists because evidence used to be requested on exactly ONE
        # frame per track - the frame JeztSort confirmed it. JeztSort confirms
        # after MIN_HITS matches, which is typically as the object ENTERS the
        # frame: smallest, most partial, most likely to fail the minimum-size
        # check. One miss there and the track never got another chance, which
        # is why tracks existed with no image at all.
        self._pending = {}
        self._captured = set()
        self._capture_lock = threading.Lock()

        self._last_failure_log = 0.0

        self._threads = [
            threading.Thread(
                target=self._worker,
                daemon=True,
                name=f"evidence-{index}",
            )
            for index in range(max(1, workers))
        ]

    def start(self):
        for thread in self._threads:
            thread.start()

    # ------------------------------------------------------- capture policy
    def needs_capture(self, camera_id, track_id):
        """
        True while this track still has no representative crop and has
        attempts left. Drives want_evidence, so a track keeps trying on
        subsequent events until one crop lands - at most once per event, and
        it stops entirely the moment a crop is submitted.
        """
        if not track_id:
            return False

        key = (camera_id, track_id)

        with self._capture_lock:
            if key in self._captured:
                return False
            return self._pending.get(key, 0) < MAX_CAPTURE_ATTEMPTS

    def note_attempt(self, camera_id, track_id, succeeded):
        """Record a capture attempt for a track."""
        if not track_id:
            return

        key = (camera_id, track_id)

        with self._capture_lock:
            if succeeded:
                self._captured.add(key)
                self._pending.pop(key, None)
                return

            attempts = self._pending.get(key, 0) + 1
            self._pending[key] = attempts

            if attempts >= MAX_CAPTURE_ATTEMPTS:
                self.exhausted += 1

    def _release_capture(self, camera_id, track_id):
        """
        An upload failed, so the track has no stored crop after all. Allow it
        to try again rather than leaving it permanently image-less.
        """
        if not track_id:
            return

        with self._capture_lock:
            self._captured.discard((camera_id, track_id))

    def forget_camera(self, camera_id):
        with self._capture_lock:
            for key in [k for k in self._captured if k[0] == camera_id]:
                self._captured.discard(key)
            for key in [k for k in self._pending if k[0] == camera_id]:
                del self._pending[key]

    def prune(self, max_tracks=20000):
        """Bound the registry on a long run."""
        with self._capture_lock:
            if len(self._captured) > max_tracks:
                self._captured.clear()
            if len(self._pending) > max_tracks:
                self._pending.clear()

    # ------------------------------------------------------------- producer
    def submit(self, jpeg, crop_width, crop_height, event, notable, camera_id,
               track_id, observed_at):
        """
        Queue a crop for upload. The event rides along and is sent by the
        worker afterwards, so it is never lost if storage misbehaves.
        Returns False only if the queue is saturated, in which case the event
        is sent immediately without evidence rather than dropped.
        """
        item = (jpeg, crop_width, crop_height, event, notable, camera_id,
                track_id, observed_at)

        try:
            self._queue.put_nowait(item)
            return True

        except Full:
            with self._lock:
                self.dropped += 1

            # Storage is backed up; the detection still matters.
            self.event_sender.submit(event, notable=notable)
            return False

    # ------------------------------------------------------------- consumer
    def _worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue

            self._process(item)

        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            self._process(item)

    def _process(self, item):
        (jpeg, crop_width, crop_height, event, notable, camera_id,
         track_id, observed_at) = item

        evidence_uid = uuid.uuid4().hex[:12]

        storage_key = build_storage_key(camera_id, track_id, observed_at, evidence_uid)

        try:
            self.store.put(storage_key, jpeg)

            with self._lock:
                self.uploaded += 1

            # Travels inside the existing metadata dict, so POST /api/events/
            # keeps its current contract - no new field, no new endpoint.
            event.setdefault("metadata", {})["evidence"] = {
                "uid": evidence_uid,
                "type": "crop",
                "storage_key": storage_key,
                "backend": self.store.name,
                "mime_type": "image/jpeg",
                "width": crop_width,
                "height": crop_height,
                "bytes": len(jpeg),
            }

        except Exception as exc:
            with self._lock:
                self.failed += 1

            # No object was stored, so this track still has no evidence.
            self._release_capture(camera_id, track_id)

            self._log_failure(f"{type(exc).__name__}: {exc}")

        # Sent either way.
        self.event_sender.submit(event, notable=notable)

    def _log_failure(self, message):
        now = time.monotonic()

        if now - self._last_failure_log < FAILURE_LOG_INTERVAL:
            return

        self._last_failure_log = now
        print(f"[EVIDENCE-ERROR] upload failed, event still sent: {message}")

    # -------------------------------------------------------------- reporting
    def pending(self):
        return self._queue.qsize()

    def stats(self):
        return {
            "uploaded": self.uploaded,
            "failed": self.failed,
            "dropped": self.dropped,
            "skipped": self.skipped,
            "exhausted": self.exhausted,
            "awaiting_capture": len(self._pending),
            "pending": self.pending(),
        }

    def note_skipped(self):
        with self._lock:
            self.skipped += 1

    def stop(self, timeout=5.0):
        self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=timeout)
