"""
Event generation and delivery to the MEERANA dashboard.

    confirmed track -> event decision -> debounce -> validate -> POST /api/events/

Three things this module is careful about:

  * DEBOUNCE. A person standing in view produces a detection on every analytic
    frame. Those are real-time track updates, not security events, and writing
    one database row per frame would bury the operator. One event per
    (camera, track, event_type) is sent, then further events for that same
    triple are suppressed for EVENT_COOLDOWN_SECONDS.

  * THE INFERENCE LOOP NEVER BLOCKS ON HTTP. Events go onto a bounded queue
    and a single background thread does the POSTing. If the dashboard is slow
    or down, the queue absorbs it; if the queue fills, the OLDEST event is
    dropped, because in a live system the newest position is the useful one.

  * NOTHING HERE CAN KILL THE PIPELINE. Every send path is wrapped; a rejected
    or malformed event is counted and logged, never raised into the loop.

Env overrides:
  EVENTS_ENABLED=0            generate and debounce but never POST (A/B testing)
  EVENT_COOLDOWN_SECONDS      per track+type suppression window (default 2.0)
  EVENT_QUEUE_SIZE            pending events held in memory     (default 2000)
  EVENT_POST_TIMEOUT          HTTP timeout in seconds           (default 3.0)
  EVENT_DEBUG=1               log every send and every suppression
"""

import os
import threading
import time
from datetime import datetime
from queue import Queue, Empty, Full

import requests

import evidence
import outbox
from tracking import PERSON_GROUP, VEHICLE_GROUP


# ============================================================
# ZAYED EVENT ABSTRACTION
#
# The reused Dubai analytics keep emitting their own event types; every event is mapped to
# the Zayed vocabulary at the one point all of them pass through - the sender's submit() -
# so no analytic module had to change. The original type rides along as
# metadata["source_event_type"]. classroom_id comes from the camera's placement (injected by
# the main loop), location / person_id from metadata when an analytic provides them.
# ============================================================

ZAYED_EVENT_TYPES = {
    "violence_detected": "FIGHT_DETECTED",
    "fall_detected": "FALL_DETECTED",
    "abandoned_object": "UNATTENDED_OBJECT_DETECTED",
    # crowd_detected is NOT mapped: it is a per-camera, per-zone person threshold, not the classroom
    # against its capacity. Mapped, one crowd raised up to one "Classroom Over Capacity" alarm per
    # camera. OVERCROWDING_DETECTED comes only from occupancy.py; zone crowds stay crowd_detected.
    "camera_tamper": "CAMERA_TAMPERING_DETECTED",
}

_classroom_resolver = None


def set_classroom_resolver(resolver):
    """Install camera_id -> classroom_id (or None). Called once by the main loop."""
    global _classroom_resolver
    _classroom_resolver = resolver


def zayed_enrich(event):
    """Map to the Zayed event vocabulary and fill the Zayed fields. Never raises."""
    try:
        metadata = event.setdefault("metadata", {})
        source = event.get("event_type")
        if source in ZAYED_EVENT_TYPES:
            metadata.setdefault("source_event_type", source)
            event["event_type"] = ZAYED_EVENT_TYPES[source]
        if not event.get("classroom_id") and _classroom_resolver is not None:
            event["classroom_id"] = _classroom_resolver(event.get("camera_id")) or ""
        if not event.get("location") and isinstance(metadata.get("location"), dict):
            event["location"] = metadata["location"]
        if not event.get("person_id"):
            event["person_id"] = str(metadata.get("known_person_external_id") or "")[:60]
        reference = metadata.get("evidence")
        if isinstance(reference, dict) and reference.get("storage_key") and not event.get("evidence_reference"):
            event["evidence_reference"] = str(reference["storage_key"])[:400]
    except Exception:                                    # noqa: BLE001 - never cost the event
        pass
    return event


# ============================================================
# CONFIGURATION
# ============================================================

EVENTS_ENABLED = os.getenv("EVENTS_ENABLED", "1") == "1"
COOLDOWN_SECONDS = float(os.getenv("EVENT_COOLDOWN_SECONDS", "2.0"))

# Zone events are debounced far harder than detections. person_detected feeds
# the Live Wall overlay, which needs a fresh box every couple of seconds; but
# every intrusion/perimeter_breach/loitering event makes the dashboard raise an
# ALARM, so a 2s cooldown would put 30 alarms in the operator's queue for one
# person standing still for a minute.
ZONE_COOLDOWN_SECONDS = float(os.getenv("ZONE_EVENT_COOLDOWN_SECONDS", "30.0"))

# ENTRY-triggered zone events need a different kind of window from presence
# ones. intrusion re-fires for as long as someone stands inside, so its
# cooldown is what stops the flood. vehicle_zone_detection cannot re-fire at
# all while the vehicle is inside - zones.py only emits it on the
# outside -> inside crossing - so the only thing a window can do here is
# suppress boundary JITTER: a bbox oscillating across the polygon edge for a
# frame or two, which would otherwise read as leave-and-return.
#
# Hence a few seconds, not thirty. Longer would start swallowing the real case
# this feature exists for: a vehicle that pulls away from the ANPR gate and a
# second one that pulls up behind it - two arrivals that both need a plate.
VEHICLE_ZONE_REENTRY_SECONDS = float(
    os.getenv("VEHICLE_ZONE_REENTRY_SECONDS", "5.0")
)

# ppe_violation is a PRESENCE condition, not an entry one: the same
# non-compliant person keeps re-triggering handle_zone_event() on every tick
# for as long as they stand in the zone, so ZONE_COOLDOWN_SECONDS's 30s
# default - built for intrusion/perimeter_breach, which want a
# near-continuous alarm - turned into "same person, same missing items,
# three Alarms in the queue within 71 seconds" on the dashboard: noise, not
# new information, for an operator. A few minutes still shows the violation
# is ONGOING without drowning the queue in repeats of a finding that has not
# changed.
PPE_VIOLATION_REENTRY_SECONDS = float(
    os.getenv("PPE_VIOLATION_REENTRY_SECONDS", "300.0")
)

# police_uniform_detected is a PRESENCE condition, exactly like
# ppe_violation above and for the same reason: the same officer standing at a
# desk re-triggers handle_uniform_event() on every tick for as long as they
# are tracked. Given the same window for the same reason - a police station
# is FULL of uniforms, so this is the analytic most able to flood a timeline,
# and repeating an unchanged observation every 30s is noise rather than new
# information. Reuses the PPE value rather than inventing a second number.
UNIFORM_DETECTED_REENTRY_SECONDS = float(
    os.getenv("UNIFORM_DETECTED_REENTRY_SECONDS",
              str(PPE_VIOLATION_REENTRY_SECONDS))
)

# Per-event-type fallbacks, consulted only when the operator has configured
# nothing for that camera. Anything absent keeps ZONE_COOLDOWN_SECONDS, so the
# intrusion / perimeter_breach / loitering / crowd windows are unchanged.
DEFAULT_EVENT_INTERVALS = {
    "vehicle_zone_detection": VEHICLE_ZONE_REENTRY_SECONDS,
    "ppe_violation": PPE_VIOLATION_REENTRY_SECONDS,
    "police_uniform_detected": UNIFORM_DETECTED_REENTRY_SECONDS,
}
QUEUE_SIZE = int(os.getenv("EVENT_QUEUE_SIZE", "2000"))
POST_TIMEOUT = float(os.getenv("EVENT_POST_TIMEOUT", "3.0"))

# One sender thread caps throughput at 1/latency - measured at ~60 ms per POST
# on the PoC rig, so ~17 events/s, which the full 19-camera estate overruns.
# A handful of threads on the same local queue lifts that past 200/s. This is
# still just threads in this process: no broker, no external state.
SENDER_THREADS = int(os.getenv("EVENT_SENDER_THREADS", "4"))
DEBUG = os.getenv("EVENT_DEBUG", "0") == "1"

# Optional floor on the confidence needed to PERSIST an event. YOLO runs at
# CONF=0.25 because weak detections still help the tracker bridge frames, but a
# 0.27-confidence "person" raising a critical perimeter_breach alarm is not
# something an operator should have to triage. Default 0.0 = no filtering, so
# behaviour is unchanged until this is deliberately raised.
MIN_CONFIDENCE = float(os.getenv("EVENT_MIN_CONFIDENCE", "0.0"))

# How often the sender is allowed to complain about the dashboard being down.
FAILURE_LOG_INTERVAL = 10.0


# Detection group -> dashboard event_type. Only these two are generated today;
# zone events are added in a later phase.
EVENT_TYPE_FOR_GROUP = {
    PERSON_GROUP: "person_detected",
    VEHICLE_GROUP: "vehicle_detected",
}


# ============================================================
# EVENT CONSTRUCTION
# ============================================================

def build_event(
    camera_id,
    event_type,
    observed_at,
    frame_width,
    frame_height,
    track_id="",
    confidence=1.0,
    bbox=None,
    metadata=None,
):
    """Low-level payload builder, shared by track and camera-level events."""
    return {
        "camera_id": camera_id,
        "event_type": event_type,
        "timestamp": datetime.fromtimestamp(observed_at).astimezone().isoformat(),
        "track_id": track_id or "",
        "confidence": round(float(confidence), 4),
        "bbox": [int(v) for v in (bbox or [])],
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "metadata": metadata or {},
    }


# ============================================================
# STABLE IDENTITY RESOLVER (layer 3)
#
# tracked.track_id is the LOGICAL TRACK identity and is not touched by any of
# this - it is consumed by zones, lines, ANPR, evidence, alarms and the
# dashboard (963 references in this tree, 377 in the dashboard), and it keeps
# its exact current meaning.
#
# metadata["stable_id"] is a SEPARATE, higher layer: the identity Re-ID
# resolved for that track, which can be the same across SEVERAL consecutive
# track_ids when a track fragmented and was recovered. It is additive - a
# free-form key in an existing JSONField, so no schema change and no
# migration - and simply ABSENT when there is no identity, which is the
# normal case for every non-Re-ID camera and every vehicle.
#
# This module deliberately imports nothing from Re-ID. multicam_inf.py
# injects the resolver at startup, so events.py has no opinion about where
# the identity comes from and no dependency on the adapter existing at all.
# ============================================================

_stable_id_resolver = None


def set_stable_id_resolver(resolver):
    """
    Install the (camera_id, track_id) -> stable_id | None callable, or None
    to disable. Called once at startup by multicam_inf.py.

    The resolver MUST be cheap and side-effect free: it is called once per
    track per emitted event.
    """
    global _stable_id_resolver
    _stable_id_resolver = resolver


def resolve_stable_id(camera_id, track_id):
    """
    The stable identity for one track, or None when there is none, no
    resolver is installed, or the resolver failed.

    A Re-ID fault must never stop an event being published, so a raising
    resolver degrades to "no stable_id" rather than propagating - the event
    still goes out, exactly as it does today with no resolver at all.
    """
    if _stable_id_resolver is None:
        return None

    try:
        return _stable_id_resolver(camera_id, track_id)
    except Exception:                                    # noqa: BLE001
        return None


def build_track_event(
    tracked,
    frame_width,
    frame_height,
    observed_at,
    event_type=None,
    extra_metadata=None,
):
    """
    Build the POST /api/events/ payload for a confirmed track.

    observed_at is the wall-clock time.time() of the frame the track was seen
    in - not "now" - so the dashboard's received_at minus timestamp is a true
    end-to-end pipeline latency.
    """
    if event_type is None:
        event_type = EVENT_TYPE_FOR_GROUP.get(tracked.group)

    metadata = {"object_type": tracked.class_name}

    # Added BEFORE extra_metadata so an explicit caller-supplied stable_id
    # still wins - the resolver is the default source, never an override.
    stable_id = resolve_stable_id(tracked.camera_id, tracked.track_id)

    if stable_id:
        metadata["stable_id"] = stable_id

    if extra_metadata:
        metadata.update(extra_metadata)

    return build_event(
        camera_id=tracked.camera_id,
        event_type=event_type,
        observed_at=observed_at,
        frame_width=frame_width,
        frame_height=frame_height,
        track_id=tracked.track_id,
        confidence=tracked.confidence,
        bbox=tracked.bbox,
        metadata=metadata,
    )


def validate_event(event, require_track=True):
    """Return an error string, or None if the payload is safe to send."""
    required = ["camera_id", "event_type", "timestamp"]

    # Zone-level events (crowd) belong to an area, not to one object, so they
    # legitimately carry no track id.
    if require_track:
        required.append("track_id")

    for field in required:
        if not event.get(field):
            return f"missing {field}"

    confidence = event.get("confidence")

    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        return f"confidence out of range: {confidence!r}"

    bbox = event.get("bbox")

    if not isinstance(bbox, list) or len(bbox) != 4:
        return f"bbox must be 4 values, got {bbox!r}"

    if not all(isinstance(v, int) for v in bbox):
        return f"bbox must be integers, got {bbox!r}"

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return f"bbox is degenerate: {bbox!r}"

    for field in ("frame_width", "frame_height"):
        value = event.get(field)

        if not isinstance(value, int) or value <= 0:
            return f"{field} must be a positive integer, got {value!r}"

    if not isinstance(event.get("metadata"), dict):
        return "metadata must be an object"

    return None


# ============================================================
# DEBOUNCE
# ============================================================

class EventDebouncer:
    """
    Suppresses repeat events for the same (camera, track, event_type).

    State is in memory and keyed on the stable track label, which is what makes
    "one event per object" possible at all - without tracking there is no way to
    tell the same person on two frames from two different people.
    """

    def __init__(self, cooldown_seconds=COOLDOWN_SECONDS):
        self.cooldown = cooldown_seconds
        self._last_sent = {}

    def allow(self, camera_id, track_id, event_type, scope=None, cooldown=None):
        # scope keeps two zones on the same camera from sharing one cooldown:
        # a track inside zone 1 and zone 2 must be able to raise both.
        key = (camera_id, track_id, event_type, scope)

        now = time.monotonic()
        last = self._last_sent.get(key)

        window = self.cooldown if cooldown is None else cooldown

        if last is not None and (now - last) < window:
            return False

        self._last_sent[key] = now
        return True

    def prune(self, max_age_seconds=300.0):
        """Forget tracks that have not been seen for a long time."""
        cutoff = time.monotonic() - max_age_seconds

        stale = [key for key, seen in self._last_sent.items() if seen < cutoff]

        for key in stale:
            del self._last_sent[key]

        return len(stale)

    def tracked_keys(self):
        return len(self._last_sent)


# ============================================================
# SENDER
# ============================================================

class EventSender:
    """Background POSTer pool. The inference loop only ever calls submit()."""

    def __init__(
        self,
        dashboard_url,
        token,
        timeout=POST_TIMEOUT,
        queue_size=QUEUE_SIZE,
        workers=SENDER_THREADS,
    ):
        self.url = f"{dashboard_url.rstrip('/')}/api/events/"
        self.token = token
        self.timeout = timeout

        self._queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()

        # Counter increments are not atomic across threads, so they are locked.
        self._lock = threading.Lock()

        self.sent = 0
        self.failed = 0
        self.dropped = 0

        self._last_failure_log = 0.0

        self._threads = [
            threading.Thread(
                target=self._worker,
                daemon=True,
                name=f"event-sender-{index}",
            )
            for index in range(max(1, workers))
        ]

    def start(self):
        for thread in self._threads:
            thread.start()

    # ------------------------------------------------------------- producer
    def submit(self, event, notable=False):
        """Non-blocking. Drops the OLDEST pending event if the queue is full."""
        try:
            self._queue.put_nowait((event, notable))
            return True

        except Full:
            try:
                self._queue.get_nowait()
                with self._lock:
                    self.dropped += 1
            except Empty:
                pass

            try:
                self._queue.put_nowait((event, notable))
                return True
            except Full:
                with self._lock:
                    self.dropped += 1
                return False

    # ------------------------------------------------------------- consumer
    def _worker(self):
        # requests.Session is not documented as thread-safe, so each worker
        # keeps its own - which also gives each its own connection pool.
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Token {self.token}",
        })

        while not self._stop_event.is_set():

            try:
                event, notable = self._queue.get(timeout=0.5)
            except Empty:
                continue

            self._post(session, event, notable)

        # Best-effort drain so events from the last second are not lost on Ctrl-C.
        while True:
            try:
                event, notable = self._queue.get_nowait()
            except Empty:
                break
            self._post(session, event, notable)

    def _post(self, session, event, notable):
        try:
            response = session.post(self.url, json=event, timeout=self.timeout)

        except Exception as exc:
            with self._lock:
                self.failed += 1
            self._log_failure(f"{type(exc).__name__}: {exc}")
            return

        if response.status_code == 201:
            with self._lock:
                self.sent += 1

            if notable or DEBUG:
                print(
                    f"[EVENT] {event['camera_id']} {event['event_type']} "
                    f"track={event['track_id']} -> 201"
                )
            return

        # The dashboard rejected it. Show why - this is a contract bug, not a
        # transient network problem, so it is worth seeing every time.
        with self._lock:
            self.failed += 1

        body = response.text[:300].replace("\n", " ")

        self._log_failure(
            f"HTTP {response.status_code} for {event['event_type']} "
            f"{event['camera_id']} track={event['track_id']}: {body}"
        )

    def _log_failure(self, message):
        now = time.monotonic()

        if now - self._last_failure_log < FAILURE_LOG_INTERVAL:
            return

        self._last_failure_log = now
        print(f"[EVENT-ERROR] {message}")

    def pending(self):
        return self._queue.qsize()

    def stop(self, timeout=5.0):
        self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=timeout)


# ============================================================
# PIPELINE
# ============================================================

class EventPipeline:
    """Glue: track -> event type -> debounce -> validate -> queue."""

    def __init__(self, dashboard_url, token, cooldown_seconds=COOLDOWN_SECONDS):
        self.enabled = EVENTS_ENABLED
        self.debouncer = EventDebouncer(cooldown_seconds)

        self.suppressed = 0
        self.invalid = 0
        self.generated = 0
        self.low_confidence = 0

        self.sender = None
        self.evidence = None

        # camera_id -> {event_type: seconds}. Populated from the dashboard
        # config and refreshed live, so an operator can slow down a noisy
        # camera without touching the others. ZONE_COOLDOWN_SECONDS remains
        # the fallback for any camera or event type not covered here.
        self._intervals = {}

        if self.enabled:
            # ZAYED: durable, retrying, idempotent delivery (outbox.py) instead of the
            # in-memory EventSender, which dropped events whenever the dashboard was down.
            self.sender = outbox.OutboxEventSender(dashboard_url, token, enrich=zayed_enrich)
            self.sender.start()

            if evidence.ENABLED:
                self.evidence = evidence.EvidencePipeline(self.sender)
                self.evidence.start()

    def handle_track(self, tracked, frame_width, frame_height, observed_at, frame=None):
        """
        Decide whether this confirmed track should become a database event.

        Feature gating happens upstream: multicam_inf.py only asks YOLO for the
        classes the camera's dashboard flags allow, so a class that reaches
        here is one the camera is configured to report.
        """
        event_type = EVENT_TYPE_FOR_GROUP.get(tracked.group)

        if event_type is None:
            return "ignored"

        # A track's first event is worth a line; the repeats after each cooldown
        # are not, or a busy camera would flood the console.
        if tracked.is_new:
            print(
                f"[{tracked.camera_id}] {tracked.class_name} detected  "
                f"track={tracked.track_id} conf={tracked.confidence:.2f}"
            )

        # One representative crop per track - but not necessarily on the
        # track's FIRST frame. is_new is true for exactly one frame, and that
        # frame is where JeztSort confirms the track: usually as the object
        # enters the scene, at its smallest and most likely to fail the
        # minimum-crop-size check. A single miss there used to leave the track
        # with no image for its entire life. needs_capture() keeps asking until
        # one crop actually lands, then stops.
        want_evidence = (
            self.evidence is not None
            and self.evidence.needs_capture(tracked.camera_id, tracked.track_id)
        )

        return self._emit(
            tracked,
            event_type,
            frame_width=frame_width,
            frame_height=frame_height,
            observed_at=observed_at,
            notable=tracked.is_new,
            frame=frame,
            want_evidence=want_evidence,
        )

    def handle_zone_event(
        self,
        tracked,
        event_type,
        zone_metadata,
        frame_width,
        frame_height,
        observed_at,
        frame=None,
    ):
        """
        Emit a zone-driven event (intrusion, perimeter_breach, loitering).

        These create Alarms in the dashboard, so each one that survives the
        debounce is logged. The zone id scopes the debounce, so a track sitting
        inside two zones raises one event per zone rather than one in total.
        """
        return self._emit(
            tracked,
            event_type,
            frame_width=frame_width,
            frame_height=frame_height,
            observed_at=observed_at,
            extra_metadata=zone_metadata,
            scope=zone_metadata.get("zone_id"),
            cooldown=self.interval_for(tracked.camera_id, event_type),
            notable=True,
            frame=frame,
            # Alarm-class events always carry a picture - it is the first thing
            # an operator wants when the queue lights up.
            want_evidence=event_type in evidence.ALARM_EVIDENCE_TYPES,
        )

    def handle_uniform_event(
        self,
        tracked,
        event_type,
        metadata,
        frame_width,
        frame_height,
        observed_at,
        frame=None,
    ):
        """
        Emit a police-uniform observation for one tracked person.

        Deliberately NOT handle_zone_event, even though the detection comes
        from the PPE model: that method hardcodes notable=True and scopes its
        debounce to a zone id, because everything routed through it is an
        alarm-class finding about a place. A uniform is neither. It raises no
        Alarm (the type is absent from the dashboard's ALARM_RULES), it
        belongs to no zone, and it asks for no still and no clip - so it
        takes the plain track path with notable=False and want_evidence off,
        which is also what keeps it out of zone de-duplication.

        Debounced per track at this type's own DEFAULT_EVENT_INTERVALS entry
        (300s), because a uniform is a presence condition rather than an
        entry one - see UNIFORM_DETECTED_REENTRY_SECONDS.

        It DOES ask for a still, through exactly the same path every other
        crop-bearing type uses: _emit crops tracked.bbox out of the frame on
        this thread and hands the bytes to the shared EvidencePipeline, which
        uploads on its own bounded queue. want_evidence is read from
        evidence.ALARM_EVIDENCE_TYPES rather than hardcoded, so there stays
        one answer to "which types get a picture".

        A still is not a clip and not an alarm. video_policy keeps this type
        at NO_VIDEO and ALARM_RULES still does not carry it; the still exists
        because "Police Uniform Detected, 53%" is unverifiable without one,
        and this class is explicitly provisional.

        The debounce is checked in _emit BEFORE the crop, so a suppressed
        repeat costs no encode and no upload - one image per track per 300s,
        not one per frame.
        """
        return self._emit(
            tracked,
            event_type,
            frame_width=frame_width,
            frame_height=frame_height,
            observed_at=observed_at,
            extra_metadata=metadata,
            cooldown=self.interval_for(tracked.camera_id, event_type),
            notable=False,
            frame=frame,
            want_evidence=event_type in evidence.ALARM_EVIDENCE_TYPES,
        )

    def handle_camera_event(
        self,
        camera_id,
        event_type,
        metadata,
        frame_width,
        frame_height,
        observed_at,
        bbox=None,
        frame=None,
        contributors=None,
        scope=None,
    ):
        """
        Emit an event that belongs to a camera/zone rather than to one track -
        crowd_detected today, camera health events later.
        """
        if not self.debouncer.allow(
            camera_id, "", event_type, scope, ZONE_COOLDOWN_SECONDS
        ):
            self.suppressed += 1
            return "suppressed"

        event = build_event(
            camera_id=camera_id,
            event_type=event_type,
            observed_at=observed_at,
            frame_width=frame_width,
            frame_height=frame_height,
            track_id="",
            confidence=1.0,
            bbox=bbox,
            metadata=metadata,
        )

        error = validate_event(event, require_track=False)

        if error is not None:
            self.invalid += 1
            print(f"[EVENT-INVALID] {camera_id} {event_type}: {error}")
            return "invalid"

        self.generated += 1

        if self.sender is None:
            return "disabled"

        if (
            frame is not None
            and bbox is not None
            and self.evidence is not None
            and event_type in evidence.ALARM_EVIDENCE_TYPES
        ):
            # The still says WHICH: the tracks the crowd count was made of
            # when the caller passed them, otherwise the subject's own box.
            # camera_tamper draws nothing - see evidence_overlay.
            cropped = evidence.crop_bbox(
                frame, bbox, overlay=(event_type, metadata, contributors))

            if cropped is not None:
                jpeg, crop_width, crop_height = cropped

                self.evidence.submit(
                    jpeg, crop_width, crop_height, event, True,
                    camera_id, f"zone{scope}" if scope else "zone", observed_at,
                )
                return "sent"

            self.evidence.note_skipped()

        self.sender.submit(event, notable=True)

        return "sent"

    def handle_abandoned_event(
        self,
        tracked,
        metadata,
        frame_width,
        frame_height,
        observed_at,
        frame=None,
    ):
        """
        Emit an abandoned_object event for an unattended bag/box.

        Like zone events this raises an Alarm in the dashboard, so it is logged,
        debounced hard, and carries an evidence crop. abandoned.py already fires
        once per resting spot; this cooldown is a backstop against a track that
        flickers in and out.
        """
        return self._emit(
            tracked,
            "abandoned_object",
            frame_width=frame_width,
            frame_height=frame_height,
            observed_at=observed_at,
            extra_metadata=metadata,
            cooldown=ZONE_COOLDOWN_SECONDS,
            notable=True,
            frame=frame,
            want_evidence="abandoned_object" in evidence.ALARM_EVIDENCE_TYPES,
        )

    def handle_object_event(
        self,
        tracked,
        metadata,
        frame_width,
        frame_height,
        observed_at,
        frame=None,
        crop_box=None,
    ):
        """
        Emit an object_detected event: one per physical object sighting.

        object_logger.py already decides that this is a new object rather than
        a fragment of one it has logged, so this is the plain track path with a
        backstop debounce. It raises no Alarm (absent from ALARM_RULES) and cuts
        no clip (NO_VIDEO); the still is the whole record of what was seen, so
        it is taken from crop_box - the object grown for context - rather than
        the tight box, which for a phone is too small to keep.
        """
        return self._emit(
            tracked,
            "object_detected",
            frame_width=frame_width,
            frame_height=frame_height,
            observed_at=observed_at,
            extra_metadata=metadata,
            notable=True,
            frame=frame,
            want_evidence="object_detected" in evidence.ALARM_EVIDENCE_TYPES,
            crop_box=crop_box,
        )

    # -------------------------------------------------------------- intervals
    def update_camera_intervals(self, camera_intervals):
        """
        Replace the per-camera event intervals.

        Called at startup and on every config refresh, so a dashboard change
        applies to the next event without restarting inference.
        """
        self._intervals = camera_intervals or {}

    def interval_for(self, camera_id, event_type):
        """
        Seconds a continuing condition must wait before re-emitting.

        Only intrusion and loitering are operator-configurable; every other
        zone event keeps its built-in default, because nothing in the dashboard
        exposes a value for them and inventing one would be a lie.
        """
        configured = self._intervals.get(camera_id, {}).get(event_type)

        if configured is None:
            return DEFAULT_EVENT_INTERVALS.get(event_type, ZONE_COOLDOWN_SECONDS)

        return configured

    # ------------------------------------------------------------------ emit
    def _emit(
        self,
        tracked,
        event_type,
        frame_width,
        frame_height,
        observed_at,
        extra_metadata=None,
        scope=None,
        cooldown=None,
        notable=False,
        frame=None,
        want_evidence=False,
        crop_box=None,
    ):
        """crop_box, when given, is cut unpadded instead of the padded track box."""
        if tracked.confidence < MIN_CONFIDENCE:
            self.low_confidence += 1
            return "low_confidence"

        if not self.debouncer.allow(
            tracked.camera_id, tracked.track_id, event_type, scope, cooldown
        ):
            self.suppressed += 1

            if DEBUG:
                print(
                    f"[DEBOUNCE] {tracked.camera_id} {event_type} "
                    f"track={tracked.track_id} suppressed"
                )
            return "suppressed"

        event = build_track_event(
            tracked,
            frame_width=frame_width,
            frame_height=frame_height,
            observed_at=observed_at,
            event_type=event_type,
            extra_metadata=extra_metadata,
        )

        error = validate_event(event)

        if error is not None:
            self.invalid += 1
            print(
                f"[EVENT-INVALID] {tracked.camera_id} {event_type} "
                f"track={tracked.track_id}: {error}"
            )
            return "invalid"

        self.generated += 1

        if self.sender is None:
            return "disabled"

        # ---- evidence ----------------------------------------------------
        # The crop happens here, on the inference thread, because this is the
        # only place the frame still exists. It is bounded by the new-track and
        # alarm rate, not the frame rate. The upload does NOT happen here.
        if want_evidence and frame is not None and self.evidence is not None:

            overlay = (event.get("event_type"), event.get("metadata") or {}, None)
            if crop_box is not None:
                cropped = evidence.crop_bbox(frame, crop_box, padding=0.0,
                                             overlay=overlay)
            else:
                cropped = evidence.crop_bbox(frame, tracked.bbox, overlay=overlay)

            # Record the outcome so a track that failed here is retried on its
            # next event, and one that succeeded is never cropped again.
            self.evidence.note_attempt(
                tracked.camera_id, tracked.track_id, cropped is not None
            )

            if cropped is not None:
                jpeg, crop_width, crop_height = cropped

                # The evidence worker uploads, attaches the reference, and then
                # sends the event itself - so ordering is upload-then-POST and
                # the dashboard never points at a missing object.
                self.evidence.submit(
                    jpeg,
                    crop_width,
                    crop_height,
                    event,
                    notable,
                    tracked.camera_id,
                    tracked.track_id,
                    observed_at,
                )

                return "sent"

            # Box too small or degenerate - send the event without a picture.
            self.evidence.note_skipped()

        self.sender.submit(event, notable=notable)

        return "sent"

    # -------------------------------------------------------------- reporting
    def stats(self):
        return {
            "generated": self.generated,
            "suppressed": self.suppressed,
            "invalid": self.invalid,
            "low_confidence": self.low_confidence,
            "sent": self.sender.sent if self.sender else 0,
            "failed": self.sender.failed if self.sender else 0,
            "dropped": self.sender.dropped if self.sender else 0,
            "pending": self.sender.pending() if self.sender else 0,
            "debounce_keys": self.debouncer.tracked_keys(),
            "evidence": self.evidence.stats() if self.evidence else None,
        }

    def prune(self):
        return self.debouncer.prune()

    def stop(self):
        # Evidence first: its workers hand events to the sender, so draining it
        # before the sender stops means nothing is stranded mid-flight.
        if self.evidence is not None:
            self.evidence.stop()

        if self.sender is not None:
            self.sender.stop()
