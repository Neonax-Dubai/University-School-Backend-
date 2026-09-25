"""
Evacuation monitoring (Zayed) - STRANDED_PERSON_DETECTED.

The backend owns the evacuation: an operator (or a fire-alarm integration) starts it with
POST /api/zayed/evacuation/start/, which creates the EvacuationSession, its deadline and the
EVACUATION_STARTED event. Inference only WATCHES:

  * a daemon thread polls GET /api/ai/evacuation/active/ every EVAC_POLL_SECONDS and keeps the
    active sessions (classroom, deadline);
  * the main loop hands every frame's confirmed person tracks to observe();
  * once a session's deadline has passed, a camera of that classroom that still sees people for
    STRANDED_CONFIRM_SECONDS raises STRANDED_PERSON_DETECTED - the evidence still is the union
    of those people's boxes - and repeats every STRANDED_REPEAT_SECONDS while they remain and
    the session is active. The backend counts them on the session.

People are counted per camera, never summed across the classroom's overlapping cameras: each
event says what ONE camera sees. A dashboard outage keeps the last known sessions for
EVAC_STALE_SECONDS, so an evacuation that is already running is not forgotten mid-way.
"""
import os
import threading
import time
from datetime import datetime

import requests

EVAC_FEATURE = "evacuation_monitoring"
EVENT_TYPE = "STRANDED_PERSON_DETECTED"
ACTIVE_PATH = "/api/ai/evacuation/active/"

EVAC_POLL_SECONDS = float(os.getenv("EVAC_POLL_SECONDS", "5"))
EVAC_STALE_SECONDS = float(os.getenv("EVAC_STALE_SECONDS", "600"))
STRANDED_CONFIRM_SECONDS = float(os.getenv("STRANDED_CONFIRM_SECONDS", "5"))
STRANDED_REPEAT_SECONDS = float(os.getenv("STRANDED_REPEAT_SECONDS", "60"))


def _epoch(iso):
    if not iso:
        return None
    return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()


class StrandedFinding:
    def __init__(self, camera_id, session, people, bbox, frame, frame_width, frame_height, observed_at):
        self.camera_id = camera_id
        self.session = session
        self.people = people
        self.bbox = bbox
        self.frame = frame
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.observed_at = observed_at

    def scope(self):
        return f"evac:{self.session['session_uid']}"

    def metadata(self):
        return {"session_uid": self.session["session_uid"], "classroom_id": self.session["classroom_id"],
                "persons_remaining": len(self.people), "track_ids": [p[0] for p in self.people],
                "deadline_at": self.session.get("deadline_at"),
                "seconds_after_deadline": round(self.observed_at - self.session["deadline"], 1),
                "method": "person_tracks_after_evacuation_deadline"}


class EvacuationMonitor:
    def __init__(self, dashboard_url, token, session=None, log=print, poll_seconds=EVAC_POLL_SECONDS):
        self.url = f"{dashboard_url.rstrip('/')}{ACTIVE_PATH}"
        self.token = token
        self._http = session
        self._log = log
        self.poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._sessions = {}           # classroom_id -> session dict (with epoch "deadline")
        self._fetched_at = None
        self._camera_classroom = {}
        self._enabled = frozenset()
        self._seen_since = {}         # (camera, session_uid) -> first time people seen after deadline
        self._last_event = {}         # (camera, session_uid) -> t
        self._stop = threading.Event()
        self._thread = None
        self.polls = self.poll_failures = self.raised = 0
        self._last_error_log = 0.0

    # ------------------------------------------------------------ configuration
    def configure(self, camera_configs):
        with self._lock:
            self._camera_classroom = {c.camera_id: (c.classroom or {}).get("classroom_id")
                                      for c in camera_configs}
            self._enabled = frozenset(c.camera_id for c in camera_configs if c.features.get(EVAC_FEATURE))

    def set_sessions(self, payload, now=None):
        """Install the /api/ai/evacuation/active/ payload (also used by tests)."""
        sessions = {}
        for s in (payload or {}).get("sessions", []):
            if s.get("status", "active") != "active" or not s.get("classroom_id"):
                continue
            deadline = _epoch(s.get("deadline_at"))
            if deadline is None:
                started = _epoch(s.get("started_at"))
                deadline = started + float(s.get("evacuation_seconds") or 180) if started else None
            if deadline is None:
                continue
            sessions[s["classroom_id"]] = dict(s, deadline=deadline)
        with self._lock:
            ended = {c: s for c, s in self._sessions.items() if c not in sessions}
            self._sessions = sessions
            self._fetched_at = now if now is not None else time.time()
        for classroom_id, s in ended.items():
            self._log(f"[EVACUATION] {classroom_id} session {s['session_uid']} ended")
        return sessions

    def active_sessions(self, now=None):
        now = now if now is not None else time.time()
        with self._lock:
            if self._fetched_at is None or now - self._fetched_at > EVAC_STALE_SECONDS:
                return {}
            return dict(self._sessions)

    # ------------------------------------------------------------ polling thread
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="evacuation-poll", daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def poll_once(self):
        if self._http is None:
            self._http = requests.Session()
            self._http.headers.update({"Authorization": f"Token {self.token}"})
        try:
            response = self._http.get(self.url, timeout=5.0)
            response.raise_for_status()
            before = set(self.active_sessions())
            sessions = self.set_sessions(response.json())
            self.polls += 1
            for classroom_id in set(sessions) - before:
                s = sessions[classroom_id]
                self._log(f"[EVACUATION] {classroom_id} session {s['session_uid']} ACTIVE - deadline "
                          f"{s.get('deadline_at')} ({s.get('evacuation_seconds')}s)")
            return True
        except Exception as exc:                          # noqa: BLE001
            self.poll_failures += 1
            now = time.monotonic()
            if now - self._last_error_log > 60:
                self._last_error_log = now
                self._log(f"[EVACUATION] poll failed ({type(exc).__name__}: {exc}) - keeping the last "
                          f"known sessions for up to {EVAC_STALE_SECONDS:.0f}s")
            return False

    def _run(self):
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.poll_seconds)

    # ------------------------------------------------------------ main loop
    def observe(self, camera_id, person_tracks, frame, frame_width, frame_height, now):
        """Returns [StrandedFinding]. Cheap when no evacuation is active."""
        if camera_id not in self._enabled:
            return []
        classroom_id = self._camera_classroom.get(camera_id)
        session = self.active_sessions(now).get(classroom_id)
        if session is None or now < session["deadline"]:
            return []
        key = (camera_id, session["session_uid"])
        if not person_tracks:
            self._seen_since.pop(key, None)
            return []
        first = self._seen_since.setdefault(key, now)
        if now - first < STRANDED_CONFIRM_SECONDS:
            return []
        last = self._last_event.get(key)
        if last is not None and now - last < STRANDED_REPEAT_SECONDS:
            return []
        self._last_event[key] = now
        boxes = [t.bbox for t in person_tracks]
        bbox = [int(min(b[0] for b in boxes)), int(min(b[1] for b in boxes)),
                int(max(b[2] for b in boxes)), int(max(b[3] for b in boxes))]
        people = [(t.track_id, list(t.bbox)) for t in person_tracks]
        self.raised += 1
        return [StrandedFinding(camera_id, session, people, bbox, frame, frame_width, frame_height, now)]

    def forget_camera(self, camera_id):
        for store in (self._seen_since, self._last_event):
            for key in [k for k in store if k[0] == camera_id]:
                del store[key]

    def stats(self):
        return {"cameras": sorted(self._enabled), "active_sessions": sorted(self.active_sessions()),
                "polls": self.polls, "poll_failures": self.poll_failures, "raised": self.raised}
