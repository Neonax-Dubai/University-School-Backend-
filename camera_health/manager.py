"""
Camera-health manager and current-state telemetry publisher.

A CPU-only SIDE PATH of the inference loop. It opens no stream, runs no model
and never touches the GPU:

    note_streams(camera_streams)        every tick: arithmetic over each reader's
                                        frames_read counter (signal loss, fps,
                                        availability)
    observe(camera_id, frame, ts)       every frame: 1 Hz per camera, reduces the
                                        already-decoded frame to a 320x180 grey
                                        sample and parks it (one slot per camera,
                                        newest wins)
    worker thread                       judges the parked samples: obstruction,
                                        scene change, defocus

CAMERA TAMPERING - one incident, one event pair
-----------------------------------------------
Only scene change (camera_tamper) and obstruction are Camera Tampering. A
camera's first tampering transition raises ONE event of type camera_tamper; a
second detector noticing the same interference joins the open incident as a
further reason; the incident recovers when its LAST condition clears.
Defocus and signal loss are diagnostics: they are detected, logged and
published in the telemetry (signal loss as status "offline", severity
"critical"), but they never open, join or close an incident (2026-09-19).

Transitions only - never a per-sample event. shadow=True evaluates everything
and emits nothing.
"""

import copy
import threading
import time
import uuid

import numpy as np

from . import detectors as D
from . import metrics
from .state import (AVAILABILITY_MIN_SLOTS, AVAILABILITY_WINDOW_SECONDS, FPS_MIN_SPAN_SECONDS,
                    FPS_STALE_SECONDS, FPS_WINDOW_SECONDS, IMAGE_STALE_SECONDS, CameraState)

FEATURE_SIGNAL_LOSS = "camera_signal_loss"
FEATURE_OBSTRUCTION = "camera_obstruction"
FEATURE_DEFOCUS = "camera_defocus"
FEATURE_TAMPER = "camera_tamper"

#: The four camera-configuration keys that arm camera health.
CAMERA_HEALTH_FEATURES = (FEATURE_SIGNAL_LOSS, FEATURE_OBSTRUCTION, FEATURE_DEFOCUS, FEATURE_TAMPER)
IMAGE_FEATURES = (FEATURE_OBSTRUCTION, FEATURE_DEFOCUS, FEATURE_TAMPER)
#: The detectors that constitute Camera Tampering.
CAMERA_TAMPERING_FEATURES = (FEATURE_TAMPER, FEATURE_OBSTRUCTION)
#: The ONE event type every tampering incident is reported as.
UMBRELLA_EVENT_TYPE = "camera_tamper"

REASON_LABELS = {
    FEATURE_TAMPER: "Scene changed",
    FEATURE_OBSTRUCTION: "View obstructed",
    FEATURE_DEFOCUS: "Lens defocused",
    FEATURE_SIGNAL_LOSS: "Signal lost",
}
SEMANTIC_NOTES = {
    FEATURE_TAMPER: "scene-change heuristic",
    FEATURE_OBSTRUCTION: "featureless-view heuristic",
}
TAMPERING_SEVERITY = "high"

SAMPLE_INTERVAL_SECONDS = 1.0
PUBLISH_INTERVAL_SECONDS = 15.0
TELEMETRY_PATH = "/api/cameras/health-telemetry/"

_CONDITION_ATTR = {
    FEATURE_SIGNAL_LOSS: "signal",
    FEATURE_OBSTRUCTION: "obstruction",
    FEATURE_DEFOCUS: "defocus",
    FEATURE_TAMPER: "tamper",
}


class _Throttle:
    """At most one log line per key per interval."""

    def __init__(self, interval=60.0):
        self.interval = interval
        self._last = {}

    def due(self, key, now=None):
        now = time.monotonic() if now is None else now
        if now - self._last.get(key, -1e18) < self.interval:
            return False
        self._last[key] = now
        return True


class CameraHealthManager:
    """Signal loss, obstruction, defocus and scene change for every armed camera."""

    def __init__(self, shadow=True, event_sink=None, logger=None):
        self.shadow = bool(shadow)
        self.event_sink = event_sink
        self._log = logger or (lambda line: print(line, flush=True))
        self._lock = threading.RLock()
        self._states = {}
        self._enabled = {}
        self._pending = {}
        self._last_accept = {}
        self._incidents = {}
        self._throttle = _Throttle()

        self.samples_taken = 0
        self.samples_dropped = 0
        self.samples_processed = 0
        self.observe_errors = 0
        self.process_errors = 0
        self.note_errors = 0
        self.sink_errors = 0
        self.transitions = 0
        self.events_emitted = 0
        self.events_shadowed = 0

        self._thread = None
        self._stop = threading.Event()
        self._wake = threading.Event()

    # ------------------------------------------------------------ configuration
    def set_enabled_cameras(self, mapping):
        """{camera_id: {feature: bool}} - replaces the armed set."""
        enabled = {}
        for camera_id, features in (mapping or {}).items():
            flags = {key: bool(value) for key, value in (features or {}).items()
                     if key in CAMERA_HEALTH_FEATURES}
            if any(flags.values()):
                enabled[str(camera_id)] = flags
        with self._lock:
            self._enabled = enabled
            for camera_id in list(self._states):
                if camera_id not in enabled:
                    self._states.pop(camera_id, None)
                    self._pending.pop(camera_id, None)
                    self._last_accept.pop(camera_id, None)
                    self._incidents.pop(camera_id, None)

    def enabled_cameras(self):
        return sorted(self._enabled)

    def _image_features(self, camera_id):
        flags = self._enabled.get(camera_id) or {}
        return {feature for feature in IMAGE_FEATURES if flags.get(feature)}

    def _state(self, camera_id, now):
        state = self._states.get(camera_id)
        if state is None:
            state = CameraState(camera_id, now)
            self._states[camera_id] = state
        return state

    # ------------------------------------------------------------ stream counters
    def note_streams(self, streams, now=None):
        """Called every tick with {camera_id: reader}; reads only frames_read."""
        now = time.time() if now is None else float(now)
        announcements = []
        with self._lock:
            for camera_id, reader in (streams or {}).items():
                flags = self._enabled.get(camera_id)
                if not flags:
                    continue
                try:
                    frames = getattr(reader, "frames_read", None)
                    if frames is None:
                        continue
                    state = self._state(camera_id, now)
                    self._note_counter(state, int(frames), now)
                    if flags.get(FEATURE_SIGNAL_LOSS):
                        gap = now - state.last_progress_at
                        action = state.signal.update(
                            gap >= D.SIGNAL_LOSS_SECONDS, now, 0.0, D.SIGNAL_LOSS_COOLDOWN_SECONDS,
                            {"gap_seconds": round(gap, 1), "threshold": D.SIGNAL_LOSS_SECONDS},
                            recover_persistence=0.0)
                        if action:
                            announcements.append((camera_id, FEATURE_SIGNAL_LOSS, action,
                                                  dict(state.signal.detail), now))
                except Exception as exc:                          # noqa: BLE001
                    self.note_errors += 1
                    if self._throttle.due(("note", camera_id)):
                        self._log(f"[CAM_HEALTH] {camera_id} note_streams error: {type(exc).__name__}: {exc}")
        for announcement in announcements:
            self._announce(*announcement)

    @staticmethod
    def _note_counter(state, frames, now):
        progressed = False
        if state.last_frames is None:
            state.last_frames = frames
            state.last_progress_at = now
        elif frames < state.last_frames:          # reader restarted its counter (reconnect)
            state.fps_window.clear()
            state.last_frames = frames
        elif frames > state.last_frames:
            state.last_frames = frames
            state.last_progress_at = now
            progressed = True
        state.last_note_at = now

        slot = int(now)
        state.slots[slot] = state.slots.get(slot, False) or progressed
        cutoff = now - AVAILABILITY_WINDOW_SECONDS
        while state.slots and next(iter(state.slots)) < cutoff:
            state.slots.popitem(last=False)

        state.fps_window.append((now, frames))
        while state.fps_window and now - state.fps_window[0][0] > FPS_WINDOW_SECONDS:
            state.fps_window.popleft()

    # ------------------------------------------------------------ image samples
    def observe(self, camera_id, frame, timestamp=None):
        """Offer one decoded frame. Returns True if it became this camera's sample.

        Never raises: a camera-health defect must never stop inference.
        """
        try:
            if not self._image_features(camera_id):
                return False
            ts = time.time() if timestamp is None else float(timestamp)
            last = self._last_accept.get(camera_id)
            if last is not None and 0.0 <= ts - last < SAMPLE_INTERVAL_SECONDS:
                return False
            if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
                self.observe_errors += 1
                return False
            height, width = frame.shape[:2]
            gray = metrics.to_sample(frame)
            with self._lock:
                state = self._state(camera_id, ts)
                size = (int(width), int(height))
                if state.frame_size is not None and state.frame_size != size:
                    state.profile_reset_pending = {"from": state.frame_size, "to": size}
                state.frame_size = size
                if camera_id in self._pending:
                    self.samples_dropped += 1
                self._pending[camera_id] = (gray, ts)
                self._last_accept[camera_id] = ts
                self.samples_taken += 1
            self._wake.set()
            return True
        except Exception as exc:                                  # noqa: BLE001
            self.observe_errors += 1
            if self._throttle.due(("observe", camera_id)):
                self._log(f"[CAM_HEALTH] {camera_id} observe error: {type(exc).__name__}: {exc}")
            return False

    def process_sample(self, camera_id, gray, t, features=None):
        """Judge one sample synchronously (the worker's path, callable directly)."""
        with self._lock:
            state = self._state(camera_id, t)
            chosen = self._image_features(camera_id) if features is None else set(features)
        return self._process(state, gray, t, chosen)

    def _process(self, state, gray, ts, features):
        features = set(features or ())
        results, announcements, messages = {}, [], []
        camera_id = state.camera_id
        with self._lock:
            gray = metrics.ensure_sample(gray)
            if state.profile_reset_pending:
                change = state.profile_reset_pending
                state.profile_reset_pending = None
                state.reset_image_baselines()
                state.profile_resets += 1
                messages.append(f"[CAM_HEALTH] {camera_id} frame size changed {change['from']} -> "
                                f"{change['to']}: image baselines reset, re-learning")

            share = metrics.structured_share(gray)
            sharp = metrics.sharpness(gray)
            state.last_sample_at = ts
            state.last_structured_share = share
            state.last_sharpness = sharp
            state.last_brightness = metrics.brightness(gray)
            obstructed = share <= D.OBSTRUCTION_EDGE_MAX

            bad, detail = D.obstruction_state(share)
            if FEATURE_OBSTRUCTION in features:
                action = state.obstruction.update(bad, ts, D.OBSTRUCTION_SECONDS,
                                                  D.OBSTRUCTION_COOLDOWN_SECONDS, detail)
                results[FEATURE_OBSTRUCTION] = (action, detail)
                if action:
                    announcements.append((camera_id, FEATURE_OBSTRUCTION, action, detail, ts))

            # Scene change before defocus: a lighting transition re-learns the
            # sharpness baseline too, and must do so before this sample is judged.
            if FEATURE_TAMPER in features:
                bad, detail = D.tamper_state(state, gray, ts, obstructed)
                if detail.get("illumination_transition"):
                    if not state.defocus.active:
                        state.defocus.since = None
                    messages.append(
                        f"[CAM_HEALTH] {camera_id} lighting transition (brightness x"
                        f"{detail.get('brightness_ratio')}, structure overlap "
                        f"{detail.get('structural_overlap')}): baselines re-learned on the new "
                        f"light - not Camera Tampering")
                if bad is not None:
                    action = state.tamper.update(bad, ts, D.TAMPER_SECONDS,
                                                 D.TAMPER_COOLDOWN_SECONDS, detail)
                    results[FEATURE_TAMPER] = (action, detail)
                    if action:
                        announcements.append((camera_id, FEATURE_TAMPER, action, detail, ts))

            bad, detail = D.defocus_state(state, sharp, obstructed)
            if FEATURE_DEFOCUS in features and bad is not None:
                action = state.defocus.update(bad, ts, D.DEFOCUS_SECONDS,
                                              D.DEFOCUS_COOLDOWN_SECONDS, detail)
                results[FEATURE_DEFOCUS] = (action, detail)
                if action:
                    announcements.append((camera_id, FEATURE_DEFOCUS, action, detail, ts))
            self.samples_processed += 1

        for message in messages:
            self._log(message)
        for announcement in announcements:
            self._announce(*announcement)
        return results

    # ------------------------------------------------------------ incidents / events
    def _announce(self, camera_id, feature, action, detail, at):
        """One detector transition. Folds tampering transitions into incidents."""
        detail = dict(detail or {})
        with self._lock:
            self.transitions += 1
        label = REASON_LABELS.get(feature, feature)
        if feature not in CAMERA_TAMPERING_FEATURES:
            self._log(f"[CAM_HEALTH] {camera_id} {label} {action} {detail} - diagnostic only, "
                      f"not Camera Tampering (no event)")
            return None

        emit = None
        with self._lock:
            incident = self._incidents.get(camera_id)
            if action == "raise":
                if incident is None:
                    incident = {"id": uuid.uuid4().hex[:12], "opened_at": at, "reason": feature,
                                "reasons": [feature], "active": {feature},
                                "conditions": {feature: detail}}
                    self._incidents[camera_id] = incident
                    emit = ("raise", self._metadata(incident, None))
                else:
                    incident["active"].add(feature)
                    if feature not in incident["reasons"]:
                        incident["reasons"].append(feature)
                    incident["conditions"][feature] = detail
                    self._log(f"[CAM_HEALTH] {camera_id} {label} joins the open incident "
                              f"{incident['id']} - no second event")
            elif action == "recover":
                if incident is None or feature not in incident["active"]:
                    return None
                incident["active"].discard(feature)
                incident["conditions"][feature] = detail
                if incident["active"]:
                    self._log(f"[CAM_HEALTH] {camera_id} {label} cleared; incident {incident['id']} "
                              f"stays open ({', '.join(sorted(incident['active']))})")
                else:
                    emit = ("recover", self._metadata(incident, at))
                    del self._incidents[camera_id]
        if emit is not None:
            self._emit(camera_id, emit[0], emit[1], at)
        return emit[0] if emit else None

    @staticmethod
    def _metadata(incident, recovered_at):
        reason = incident["reason"]
        meta = {
            "reason": reason,
            "reason_label": REASON_LABELS.get(reason, reason),
            "reasons": list(incident["reasons"]),
            "reason_labels": [REASON_LABELS.get(r, r) for r in incident["reasons"]],
            "conditions": copy.deepcopy(incident["conditions"]),
            "incident_id": incident["id"],
            "severity": TAMPERING_SEVERITY,
            "semantic_note": SEMANTIC_NOTES.get(reason),
            "opened_at": incident["opened_at"],
        }
        if recovered_at is not None:
            meta["recovered"] = True
            meta["duration_seconds"] = round(recovered_at - incident["opened_at"], 3)
        return meta

    def _emit(self, camera_id, action, metadata, at):
        state = self._states.get(camera_id)
        width, height = (state.frame_size if state is not None and state.frame_size else (None, None))
        if self.shadow:
            self.events_shadowed += 1
            self._log(f"[CAM_HEALTH][SHADOW] {camera_id} would {action} {UMBRELLA_EVENT_TYPE} "
                      f"({metadata.get('reason_label')}) incident={metadata.get('incident_id')}")
            return
        if self.event_sink is None:
            return
        try:
            self.event_sink(camera_id=camera_id, event_type=UMBRELLA_EVENT_TYPE, timestamp=at,
                            metadata=metadata, action=action, frame_width=width, frame_height=height)
            self.events_emitted += 1
        except Exception as exc:                                  # noqa: BLE001
            self.sink_errors += 1
            if self._throttle.due(("sink", camera_id)):
                self._log(f"[CAM_HEALTH] {camera_id} event sink error: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ worker
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-health", daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait(1.0)
            self._wake.clear()
            with self._lock:
                items = list(self._pending.items())
                self._pending.clear()
            for camera_id, (gray, ts) in items:
                try:
                    with self._lock:
                        state = self._states.get(camera_id)
                        features = self._image_features(camera_id)
                    if state is not None:
                        self._process(state, gray, ts, features)
                except Exception as exc:                          # noqa: BLE001
                    self.process_errors += 1
                    if self._throttle.due(("process", camera_id)):
                        self._log(f"[CAM_HEALTH] {camera_id} worker error: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ current state
    def snapshot(self, now=None):
        """Plain-dict current state of every armed camera. Emits nothing."""
        now = time.time() if now is None else float(now)
        with self._lock:
            return [self._row(camera_id, self._states.get(camera_id), now)
                    for camera_id in sorted(self._enabled)]

    def _row(self, camera_id, state, now):
        row = {"camera_id": camera_id, "status": None, "severity": None, "telemetry": "none",
               "signal": None, "issue": None, "conditions": [], "availability_pct": None,
               "actual_fps": None, "blur_score": None, "obstruction_pct": None,
               "scene_distance": None, "baseline_phase": None, "incident_id": None,
               "incident_reason": None, "observed_at": round(now, 3),
               # the dashboard anchors freshness on measured_at (epoch seconds)
               "measured_at": round(now, 3)}
        if state is None or not state.observed():
            return row

        flags = self._enabled.get(camera_id) or {}
        availability = self._availability(state, now)
        fps = self._fps(state, now)
        fresh = state.last_sample_at is not None and now - state.last_sample_at <= IMAGE_STALE_SECONDS
        blur = None
        if fresh and state.sharpness_baseline and state.last_sharpness is not None:
            blur = round(100.0 * state.last_sharpness / state.sharpness_baseline, 1)
        obstruction = None
        if fresh and state.last_structured_share is not None:
            obstruction = round(100.0 * (1.0 - state.last_structured_share), 1)

        signal, lost = None, False
        if state.last_note_at is not None:
            lost = state.signal.active if flags.get(FEATURE_SIGNAL_LOSS) else False
            lost = lost or (now - state.last_progress_at >= D.SIGNAL_LOSS_SECONDS)
            signal = "lost" if lost else "receiving"

        active = [f for f in IMAGE_FEATURES if getattr(state, _CONDITION_ATTR[f]).active]
        if lost:
            status, severity, issue = "offline", "critical", REASON_LABELS[FEATURE_SIGNAL_LOSS]
        elif active:
            status = "warning"
            issue = ", ".join(REASON_LABELS[f] for f in active)
            severity = TAMPERING_SEVERITY if any(f in CAMERA_TAMPERING_FEATURES for f in active) else "medium"
        else:
            status, severity, issue = "healthy", "info", None

        incident = self._incidents.get(camera_id)
        row.update({
            "status": status, "severity": severity,
            "telemetry": "live" if availability is not None else "warming_up",
            "signal": signal, "issue": issue,
            "conditions": ([FEATURE_SIGNAL_LOSS] if lost else []) + active,
            "availability_pct": availability, "actual_fps": fps,
            "blur_score": blur, "obstruction_pct": obstruction,
            "scene_distance": (round(state.last_distance, 4)
                               if fresh and state.last_distance is not None else None),
            "baseline_phase": state.scene_phase,
            "incident_id": incident["id"] if incident else None,
            "incident_reason": incident["reason"] if incident else None,
        })
        return row

    @staticmethod
    def _availability(state, now):
        cutoff = now - AVAILABILITY_WINDOW_SECONDS
        slots = [up for second, up in state.slots.items() if cutoff <= second <= now]
        if len(slots) < AVAILABILITY_MIN_SLOTS:
            return None
        return round(100.0 * sum(slots) / len(slots), 1)

    @staticmethod
    def _fps(state, now):
        if state.last_note_at is None or now - state.last_note_at > FPS_STALE_SECONDS:
            return None
        window = state.fps_window
        if len(window) < 2:
            return None
        (t0, f0), (t1, f1) = window[0], window[-1]
        span = t1 - t0
        if span < FPS_MIN_SPAN_SECONDS:
            return None
        return round((f1 - f0) / span, 1)

    # ------------------------------------------------------------ reporting
    def stats(self):
        with self._lock:
            return {
                "shadow": self.shadow, "cameras": len(self._enabled), "tracked": len(self._states),
                "samples_taken": self.samples_taken, "samples_dropped": self.samples_dropped,
                "samples_processed": self.samples_processed, "queue_depth": len(self._pending),
                "queue_max_per_camera": 1, "transitions": self.transitions,
                "events_emitted": self.events_emitted, "events_shadowed": self.events_shadowed,
                "open_incidents": len(self._incidents), "observe_errors": self.observe_errors,
                "process_errors": self.process_errors, "note_errors": self.note_errors,
                "sink_errors": self.sink_errors,
            }

    def status_line(self):
        s = self.stats()
        return (f"[CAM_HEALTH] mode={'shadow' if s['shadow'] else 'live'} cameras={s['cameras']} "
                f"samples={s['samples_processed']} dropped={s['samples_dropped']} "
                f"transitions={s['transitions']} events={s['events_emitted']} "
                f"open_incidents={s['open_incidents']} errors="
                f"{s['observe_errors'] + s['process_errors'] + s['note_errors'] + s['sink_errors']}")


class HealthTelemetryPublisher:
    """Posts the whole estate's current state, one bulk request every interval.

    Never queues: a failed cycle is forgotten and the next one carries fresh
    state. Runs on its own daemon thread; the frame path never waits for it.
    """

    def __init__(self, manager, dashboard_url, token, session=None, log=None,
                 interval=PUBLISH_INTERVAL_SECONDS, timeout=5.0, path=TELEMETRY_PATH):
        self.manager = manager
        self.url = f"{(dashboard_url or '').rstrip('/')}{path}"
        self.token = token
        self.interval = float(interval)
        self.timeout = float(timeout)
        self._session = session
        self._log = log or (lambda *parts: print(*parts, flush=True))
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._throttle = _Throttle()
        self._stats = {"sent": 0, "failed": 0, "errors": 0, "skipped_empty": 0,
                       "last_status": None, "last_ok_at": None, "last_error": None}

    def _bump(self, key, **extra):
        with self._lock:
            self._stats[key] += 1
            self._stats.update(extra)

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        headers = getattr(self._session, "headers", None)
        if headers is not None and self.token:
            headers.update({"Authorization": f"Token {self.token}"})
        return self._session

    def publish_once(self):
        try:
            rows = self.manager.snapshot(now=time.time())
        except Exception as exc:                                  # noqa: BLE001
            self._bump("errors", last_error=f"snapshot: {type(exc).__name__}: {exc}")
            return False
        if not rows:
            self._bump("skipped_empty")
            return False
        payload = {"cameras": rows, "published_at": time.time()}
        try:
            response = self._http().post(self.url, json=payload, timeout=self.timeout)
        except Exception as exc:                                  # noqa: BLE001
            self._bump("failed", last_error=f"{type(exc).__name__}: {exc}")
            if self._throttle.due("post"):
                self._log(f"[CAM_HEALTH] telemetry publish failed: {type(exc).__name__}: {exc}")
            return False
        code = getattr(response, "status_code", 0)
        if 200 <= code < 300:
            self._bump("sent", last_status=code, last_ok_at=time.time())
            return True
        text = str(getattr(response, "text", ""))[:160]
        self._bump("failed", last_status=code, last_error=f"HTTP {code}: {text}")
        if self._throttle.due("http"):
            self._log(f"[CAM_HEALTH] telemetry publish rejected: HTTP {code}: {text}")
        return False

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-health-telemetry", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.publish_once()
            except Exception as exc:                              # noqa: BLE001
                self._bump("errors", last_error=f"{type(exc).__name__}: {exc}")

    def stop(self, timeout=5.0):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    def stats(self):
        with self._lock:
            return dict(self._stats)

    def status_line(self):
        s = self.stats()
        return (f"[CAM_HEALTH-TELEMETRY] sent={s['sent']} failed={s['failed']} errors={s['errors']} "
                f"skipped_empty={s['skipped_empty']} last_status={s['last_status']}")
