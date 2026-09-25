"""
Dashboard client - camera configuration from the MEERANA dashboard.

The Django dashboard is the SOURCE OF TRUTH for camera configuration. This
module pulls the AI-enabled camera list from GET /api/ai/cameras/ so that no
camera_id, stream URL or feature flag is ever hardcoded in the pipeline.

Stream routing
--------------
The dashboard stores each camera's MAIN stream (full resolution). Analytics run
on the MAIN stream (best accuracy for small objects - PPE, ANPR plates), served
by MediaMTX at :18554/<camera_id>. The operator live wall reads the separate
"-web" path, which MediaMTX transcodes from the lighter SUB stream - so the wall
stays light while inference gets full resolution.

So the camera LIST comes from the dashboard, and the inference stream URL is
resolved through MediaMTX by camera_id:

    CAM-R01  ->  rtsp://10.232.7.151:18554/cam-r01        (MAIN, for AI)
    wall     ->  http://10.232.7.151:8889/cam-r01-web     (SUB->H264, for browser)

The dashboard's rtsp_url is the same main stream, carried on the config object
for reference and evidence capture.
"""

import json
import os
import threading
import time

import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env_file(path):
    """Populate os.environ from a .env file (avoids a python-dotenv dependency)."""
    try:
        with open(path) as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")

        # Real environment always wins over the file.
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file(os.path.join(BASE_DIR, ".env"))


DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://zayed-dashboard:8000")
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "")

# Base of the MediaMTX RTSP fan-out. Each camera is served at "<base>/<camera_id lowered>".
MEDIAMTX_BASE_URL = os.getenv("MEDIAMTX_BASE_URL", "rtsp://mediamtx:8554")

FETCH_TIMEOUT = 10.0

# Last good configuration, so a dashboard outage cannot stop the pipeline starting.
# ZAYED: last-known-good camera configuration, written by this process under runtime/ (never in
# git). It carries no credential: stream sources are local, credential-free MediaMTX URLs.
CACHE_FILE = os.getenv("CAMERA_CONFIG_CACHE", os.path.join(BASE_DIR, "runtime", "cameras.cache.json"))


# Features the pipeline actually acts on today. Everything else the dashboard
# sends is carried through and reported as pending, so an operator can never
# mistake a flag that is merely switched on for an analytic that is running.
IMPLEMENTED_FEATURES = {
    "person_detection",
    "vehicle_detection",
    "intrusion",
    "perimeter_breach",
    "loitering",
    "line_crossing",
    "crowd_detection",
    "abandoned_object",
    "ppe_compliance",
    "behaviour_analysis",
    "physical_distancing",
    "face_detection",
    "fall_detection",
    "fire_smoke_detection",
    "violence_detection",
    "fence_climbing_detection",
    # Camera health. Added when these went live: while they ran in shadow the
    # banner correctly reported them as pending, and leaving them out now would
    # make it claim "not implemented yet" about analytics that are creating
    # production events.
    "camera_tamper",
    "camera_obstruction",
    "camera_defocus",
    "camera_signal_loss",
    "people_counting",
    "object_detection",
    # ZAYED classroom analytics
    "sleeping_detection",
    "mobile_phone_detection",
    "evacuation_monitoring",
    "occupancy_heatmap",
}

#: Features this process deliberately does NOT run, because another process
#: does. Reported separately from "pending": ANPR is fully implemented, it just
#: lives in the ANPR worker, which reads the vehicle crop this process already
#: stored. Calling it "not implemented yet" in the startup banner would send an
#: operator looking for a bug that is not there.
ELSEWHERE_FEATURES = {
    "anpr": "ANPR worker",
    # Its own process (weapon_detection/), started separately from
    # run_inference.py; this process never loads the weapon model.
    "weapon_detection": "weapon worker",
}


# Fallbacks for a payload that predates the camera-level parameters, or a
# cached config written by an older dashboard. They match the inference
# constants they replace, so behaviour is unchanged when the field is absent.
DEFAULT_DETECTION_CONFIDENCE = float(os.getenv("DEFAULT_DETECTION_CONFIDENCE", "0.45"))
DEFAULT_EVENT_INTERVAL = float(os.getenv("ZONE_EVENT_COOLDOWN_SECONDS", "30.0"))


def _float_or(value, fallback, low, high):
    """Coerce a config value, falling back when absent or out of range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback

    return number if low <= number <= high else fallback


class DashboardError(RuntimeError):
    """The camera configuration could not be loaded."""


# ============================================================
# CAMERA CONFIG
# ============================================================

class CameraConfig:
    """One AI-enabled camera as configured in the dashboard."""

    def __init__(
        self,
        camera_id,
        name,
        source,
        rtsp_url,
        width,
        height,
        features,
        zones,
        lines,
        detection_confidence,
        intrusion_interval,
        loitering_interval,
        calibration=None,
        fences=None,
        object_classes=None,
        classroom=None,
        placement=None,
    ):
        # ZAYED: which classroom this camera watches (id, capacity, floor plan, use cases) and
        # how it is placed (MediaMTX path, floor-plan homography) - from the backend.
        self.classroom = classroom or {}
        self.placement = placement or {}
        self.camera_id = camera_id
        self.name = name
        self.source = source          # MediaMTX MAIN stream, what we infer on
        self.rtsp_url = rtsp_url      # same main stream direct from the dashboard
        self.width = width
        self.height = height
        self.features = features
        self.zones = zones
        # Ground-plane calibration for physical distancing, straight from the
        # dashboard. Optional so an older payload still constructs.
        self.calibration = calibration or {}
        self.lines = lines
        # Fence segments, normalised 0-1 exactly like `lines`. Optional with a
        # list default so a payload from a dashboard that predates fences still
        # constructs - the same tolerance `calibration` above already has.
        self.fences = fences or []
        # COCO class names object detection logs on this camera. Empty means
        # object_logger's defaults; a payload from an older dashboard has none.
        self.object_classes = list(object_classes or [])

        # Camera-level analytic parameters. The dashboard is the source of
        # truth; these defaults only apply when a payload predates the fields.
        self.detection_confidence = detection_confidence
        self.intrusion_interval = intrusion_interval
        self.loitering_interval = loitering_interval

    def enabled_features(self):
        """Feature names switched on in the dashboard."""
        return sorted(
            name
            for name, on in self.features.items()
            if on
        )

    def active_features(self):
        """Enabled features the pipeline implements today."""
        return [
            name
            for name in self.enabled_features()
            if name in IMPLEMENTED_FEATURES
        ]

    def pending_features(self):
        """Enabled features nothing implements yet - genuinely unbuilt."""
        return [
            name
            for name in self.enabled_features()
            if name not in IMPLEMENTED_FEATURES
            and name not in ELSEWHERE_FEATURES
        ]

    def elsewhere_features(self):
        """Enabled features handled by another process, and which one."""
        return [
            (name, ELSEWHERE_FEATURES[name])
            for name in self.enabled_features()
            if name in ELSEWHERE_FEATURES
        ]

    def __repr__(self):
        return f"<CameraConfig {self.camera_id} {self.source}>"


# ============================================================
# FETCH
# ============================================================

def mediamtx_source(camera_id):
    """CAM-R01 -> rtsp://<mediamtx>/cam-r01"""
    return f"{MEDIAMTX_BASE_URL.rstrip('/')}/{camera_id.lower()}"


def _parse(payload):
    """Turn the API payload into CameraConfig objects."""
    cameras = []

    for entry in payload.get("cameras", []):

        # The endpoint already filters on ai_enabled; check anyway so a change
        # there can never silently start inferring on a disabled camera.
        if not entry.get("ai_enabled", False):
            continue

        camera_id = entry.get("camera_id")

        if not camera_id:
            continue

        resolution = entry.get("resolution") or {}

        cameras.append(
            CameraConfig(
                camera_id=camera_id,
                name=entry.get("name") or camera_id,
                # ZAYED: the backend's stream_url (from the classroom placement) is the source
                # of truth; the camera-id convention is only a fallback.
                source=entry.get("stream_url") or mediamtx_source(camera_id),
                rtsp_url=entry.get("rtsp_url", ""),
                width=int(resolution.get("width", 1920)),
                height=int(resolution.get("height", 1080)),
                features=entry.get("features") or {},
                zones=entry.get("zones") or [],
                lines=entry.get("lines") or [],
                fences=entry.get("fences") or [],
                detection_confidence=_float_or(
                    (entry.get("detection") or {}).get("confidence"),
                    DEFAULT_DETECTION_CONFIDENCE, 0.0, 1.0,
                ),
                intrusion_interval=_float_or(
                    (entry.get("intervals") or {}).get("intrusion"),
                    DEFAULT_EVENT_INTERVAL, 1.0, 3600.0,
                ),
                loitering_interval=_float_or(
                    (entry.get("intervals") or {}).get("loitering"),
                    DEFAULT_EVENT_INTERVAL, 1.0, 3600.0,
                ),
                calibration=entry.get("calibration") or {},
                object_classes=(entry.get("objects") or {}).get("classes") or [],
                classroom=entry.get("classroom") or {},
                placement=entry.get("placement") or {},
            )
        )

    cameras.sort(key=lambda camera: camera.camera_id)

    return cameras


def _save_cache(payload):
    try:
        with open(CACHE_FILE, "w") as handle:
            json.dump(payload, handle, indent=2)
    except OSError as exc:
        print(f"[dashboard] Warning: could not write {CACHE_FILE}: {exc}")


def _load_cache():
    """Return (payload, age_seconds) for the cached config, or (None, None)."""
    try:
        age = time.time() - os.path.getmtime(CACHE_FILE)

        with open(CACHE_FILE) as handle:
            return json.load(handle), age

    except (OSError, ValueError):
        return None, None


def _format_age(seconds):
    if seconds < 90:
        return f"{seconds:.0f} seconds"

    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"

    return f"{seconds / 3600:.1f} hours"


def fetch_cameras(use_cache_on_failure=True):
    """
    Return the AI-enabled cameras configured in the dashboard.

    Raises DashboardError if the dashboard is unreachable and no cached
    configuration is available.
    """
    url = f"{DASHBOARD_URL.rstrip('/')}/api/ai/cameras/"

    if not DASHBOARD_TOKEN:
        raise DashboardError(
            "DASHBOARD_TOKEN is not set. Add it to AI_inferencing/.env - get one with:\n"
            f"  curl -X POST {DASHBOARD_URL}/api/auth/token/ "
            '-H "Content-Type: application/json" '
            "-d '{\"username\": \"admin\", \"password\": \"...\"}'"
        )

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Token {DASHBOARD_TOKEN}"},
            timeout=FETCH_TIMEOUT,
        )

        if response.status_code == 401:
            raise DashboardError(
                f"Dashboard rejected the token (401) at {url}. "
                "Check DASHBOARD_TOKEN in AI_inferencing/.env"
            )

        response.raise_for_status()

        payload = response.json()

    except DashboardError:
        raise

    except Exception as exc:

        if use_cache_on_failure:
            cached, age = _load_cache()

            if cached is not None:
                camera_count = len(cached.get("cameras", []))

                print("")
                print("!" * 68)
                print("! WARNING: Dashboard unavailable.")
                print("!          Using cached camera configuration.")
                print(f"!          Cache age: {_format_age(age)} ({age:.0f} seconds)")
                print("!" * 68)
                print(f"!   endpoint : {url}")
                print(f"!   reason   : {exc}")
                print(f"!   cache    : {CACHE_FILE}")
                print(f"!   cameras  : {camera_count} (from cache)")
                print("!   Camera list, feature flags and zones may be STALE.")
                print("!" * 68)
                print("")
                return _parse(cached)

        raise DashboardError(f"Cannot reach {url}: {exc}") from exc

    _save_cache(payload)

    return _parse(payload)


# ============================================================
# BACKGROUND CONFIG WATCHER
# ============================================================
#
# The dashboard stays the source of truth WHILE the pipeline runs: flip a
# feature or draw a zone in the UI and the change is picked up without a
# restart.
#
# This thread only performs the HTTP fetch. It never touches pipeline state -
# it parks the result and the main loop collects it between batches with
# take(). Camera streams, feature maps and zone state are therefore only ever
# mutated from the inference thread, so no locking is needed around them.

REFRESH_SECONDS = float(os.getenv("CONFIG_REFRESH_SECONDS", "30"))


class ConfigWatcher(threading.Thread):
    """Polls GET /api/ai/cameras/ and parks the newest result for the main loop."""

    def __init__(self, interval=REFRESH_SECONDS):
        super().__init__(daemon=True, name="config-watcher")

        self.interval = interval

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest = None

        self.refreshes = 0
        self.failures = 0
        self.last_success = None

        self._last_failure_log = 0.0

    def run(self):
        # wait() returns True only when stopped, so this both paces the loop
        # and exits promptly on shutdown.
        while not self._stop_event.wait(self.interval):

            try:
                # No cache fallback here: if the dashboard is unreachable we
                # keep running the config we already have rather than swapping
                # a live configuration for a stale file.
                cameras = fetch_cameras(use_cache_on_failure=False)

            except Exception as exc:
                self.failures += 1

                now = time.monotonic()

                if now - self._last_failure_log > 30.0:
                    self._last_failure_log = now
                    print(f"[CONFIG] refresh failed, keeping current config: {exc}")

                continue

            with self._lock:
                self._latest = cameras
                self.refreshes += 1
                self.last_success = time.time()

    def take(self):
        """Newest config fetched since the last call, or None if nothing new."""
        with self._lock:
            cameras, self._latest = self._latest, None
            return cameras

    def age(self):
        """Seconds since the last successful fetch, or None."""
        if self.last_success is None:
            return None
        return time.time() - self.last_success

    def stop(self):
        self._stop_event.set()
