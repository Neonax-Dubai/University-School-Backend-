"""
Re-ID camera-group configuration provider (Phase 2A + production-hardening
P0 fix).

Keeps reid_poc.config.REID_CAMERA_GROUPS AND config.REID_ENABLED_CAMERAS in
sync with the dashboard's Re-ID policy (GET /api/ai/reid-groups/), while the
reid_poc/ PoC's own static JSON/inline config remains the initial value and
the ultimate fallback - never deleted, never bypassed, only ever REPLACED
once a valid dashboard config is available.

REID_ENABLED_CAMERAS is the dashboard's Re-ID AUTHORIZATION set (Camera.
reid_enabled, regardless of group membership - see cameras/api_views.py's
_reid_enabled_cameras_payload()) - added alongside REID_CAMERA_GROUPS for the
production-hardening P0 fix (group isolation could previously be bypassed by
any camera the dashboard had not explicitly configured; see reid_manager.py's
needs_observation() for the actual enforcement point this config feeds).

    Startup:  reid_poc/config.py has already loaded its static config at
              import time (unconditionally, before this module ever runs) ->
              ReIDConfigProvider.__init__() makes ONE synchronous, bounded-
              timeout dashboard fetch -> a VALID response replaces the static
              config; anything else (unreachable, malformed, invalid) leaves
              the static config in place, untouched.
    Runtime:  a background daemon thread re-fetches every
              REID_CONFIG_REFRESH_SECONDS and, only on a successful fetch +
              validation, atomically replaces the live config with the new
              one. A failed refresh NEVER erases the last known good config -
              it is simply skipped, with a rate-limited warning logged.

Atomicity: config.REID_CAMERA_GROUPS and config.REID_ENABLED_CAMERAS are each
replaced by ONE module-attribute rebinding (config.REID_CAMERA_GROUPS =
new_dict / config.REID_ENABLED_CAMERAS = new_frozenset), never mutated in
place (no .clear()/.update()/per-key writes, no .add()/.discard()). CPython's
GIL makes a single attribute rebinding atomic with respect to any other
thread reading it, and every existing reader (resolve_camera_group(),
GlobalReIDManager._policy_for()/needs_observation()/observe()) only ever
does a dict/set lookup against ONE captured module reference per call - so
an in-flight Re-ID decision always sees either the complete old value or the
complete new one, never a partial one. The two attributes are always
validated together and rebound back-to-back in _fetch_and_apply() (groups
first, then enabled-cameras) - see that method's own docstring for why a
reader racing between the two rebindings can only ever observe a MORE
restrictive combination, never a more permissive one.

Reuses dashboard.py's DASHBOARD_URL/DASHBOARD_TOKEN (imported, never
duplicated) and its exact Token-auth convention - the SAME dashboard
credential multicam_inf.py's own camera-config fetch already uses.

This is a periodic background poll (REID_CONFIG_REFRESH_SECONDS, default
30s), never a per-frame or per-track request - process_tracks()/flush() in
reid_adapter.py never call anything in this module.
"""
import os
import threading
import time

import requests

import config      # reid_poc/config.py - already on sys.path by the time
                   # this module is imported (see reid_adapter.py's
                   # _init_reid_poc(), the only caller)
import dashboard   # production, UNMODIFIED - DASHBOARD_URL/DASHBOARD_TOKEN only


REID_CONFIG_REFRESH_SECONDS = float(os.getenv("REID_CONFIG_REFRESH_SECONDS", "30.0"))
REID_CONFIG_FETCH_TIMEOUT_SECONDS = float(os.getenv("REID_CONFIG_FETCH_TIMEOUT_SECONDS", "5.0"))

_ENDPOINT_PATH = "/api/ai/reid-groups/"

# Rate-limit warning spam exactly like reid_adapter.py's own _log_error - a
# dashboard outage must produce one clear warning periodically, not one line
# per failed poll.
_WARN_LOG_INTERVAL_SECONDS = 30.0


def _log(message):
    print(f"[REID-CONFIG] {message}")


def _describe_group(name, cfg):
    threshold = "default" if cfg["similarity_threshold"] is None else f"{cfg['similarity_threshold']:.2f}"
    margin = "default" if cfg["min_match_margin"] is None else f"{cfg['min_match_margin']:.2f}"
    mode = cfg["camera_mode"] or "default"
    for camera_id in cfg["cameras"]:
        _log(f"camera={camera_id} group={name} threshold={threshold} margin={margin} mode={mode}")


def fetch_reid_groups():
    """
    One GET /api/ai/reid-groups/ call. Returns (groups, raw_enabled_cameras)
    on HTTP 200 with valid JSON:
      - groups: the raw {group_name: {...}} dict (required - missing/wrong
        type is malformed, exactly as before this field existed).
      - raw_enabled_cameras: payload.get("reid_enabled_cameras") - the raw
        list, or None when the key is absent/null. None is NOT treated as
        malformed here (an older dashboard build predating the P0 fix would
        omit it) - the caller falls back to config._default_enabled_cameras(),
        the SAME safe fallback reid_poc/config.py's own static-JSON loader
        uses when ITS file lacks the key (see config._load_enabled_cameras()).
        A key that IS present but the wrong shape, or inconsistent with the
        fetched groups, is NOT defaulted - config._validate_and_normalize_
        enabled_cameras() raises for that, exactly like any other malformed
        field (see _fetch_and_apply()).

    Raises on anything else (network error, non-200, malformed JSON) - the
    caller decides what "raises" means for the live configuration (see
    ReIDConfigProvider._fetch_and_apply()); this function never touches
    config.REID_CAMERA_GROUPS/REID_ENABLED_CAMERAS itself.
    """
    if not dashboard.DASHBOARD_TOKEN:
        raise RuntimeError("DASHBOARD_TOKEN is not set")

    url = f"{dashboard.DASHBOARD_URL.rstrip('/')}{_ENDPOINT_PATH}"
    response = requests.get(
        url,
        headers={"Authorization": f"Token {dashboard.DASHBOARD_TOKEN}"},
        timeout=REID_CONFIG_FETCH_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), dict):
        raise ValueError(f"malformed response from {url}: expected {{'groups': {{...}}}}")

    return payload["groups"], payload.get("reid_enabled_cameras")


class ReIDConfigProvider:
    """
    Owns the startup fetch + background refresh thread. One instance per
    ReIDAdapter, created (which performs the synchronous startup fetch) and
    started only from _init_reid_poc() - i.e. only when Re-ID is enabled -
    and stopped from ReIDAdapter.close().
    """

    def __init__(self, interval=None, timeout=None):
        self.interval = REID_CONFIG_REFRESH_SECONDS if interval is None else interval
        self.timeout = REID_CONFIG_FETCH_TIMEOUT_SECONDS if timeout is None else timeout

        self.refreshes = 0
        self.failures = 0
        self.last_success = None

        self._stop_event = threading.Event()
        self._last_warn_log = 0.0
        self._thread = None

        # Startup fetch: SYNCHRONOUS, bounded by self.timeout (requests' own
        # timeout already bounds this call - no extra wrapping needed). Runs
        # here, before start() launches the background thread, so "static
        # loaded -> dashboard fetched -> valid config replaces static"
        # happens in that exact order, once, at construction time.
        self._fetch_and_apply(context="startup")

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="reid-config-watcher")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        # Same shape as dashboard.ConfigWatcher.run(): wait() returns True
        # only when stopped, so this both paces the loop and exits promptly
        # on shutdown.
        while not self._stop_event.wait(self.interval):
            self._fetch_and_apply(context="refresh")

    def _fetch_and_apply(self, context):
        """
        Fetch, validate, and - only if BOTH pieces validate successfully -
        atomically replace config.REID_CAMERA_GROUPS AND
        config.REID_ENABLED_CAMERAS together. Never raises: a failure here
        must never take down the calling thread (the constructor, or the
        background watcher loop), it only means "try again later, current
        config stays live" - see the module docstring's fallback semantics.

        Both normalized values are computed BEFORE either is applied, so a
        problem with either one (malformed groups, malformed/inconsistent
        enabled-cameras) rejects the WHOLE fetch and leaves BOTH config
        attributes at their last-known-good values - never a partial update
        of one without the other. This is what makes "a configuration
        failure must never broaden the candidate search space" hold for the
        authorization set exactly as it already held for the groups.
        """
        try:
            raw_groups, raw_enabled_cameras = fetch_reid_groups()
            normalized_groups = config._validate_and_normalize_camera_groups(
                raw_groups, config.REID_DEFAULT_GROUP
            )
            if raw_enabled_cameras is None:
                # Dashboard omitted the field (older build) - fall back to
                # the same safe default reid_poc/config.py itself uses when
                # its static file lacks the key: the union of the groups'
                # own camera lists, never wider.
                normalized_enabled = config._default_enabled_cameras(normalized_groups)
            else:
                normalized_enabled = config._validate_and_normalize_enabled_cameras(
                    raw_enabled_cameras, normalized_groups
                )
        except Exception as exc:  # noqa: BLE001 - must never propagate
            self.failures += 1
            now = time.monotonic()
            if now - self._last_warn_log > _WARN_LOG_INTERVAL_SECONDS:
                self._last_warn_log = now
                _log(f"WARNING dashboard config unavailable or invalid ({context}), "
                     f"keeping current configuration: {type(exc).__name__}: {exc}")
            return

        changed = (
            normalized_groups != config.REID_CAMERA_GROUPS
            or normalized_enabled != config.REID_ENABLED_CAMERAS
        )

        # THE atomic replacement: two sequential name rebindings, never
        # in-place mutation (no .clear()/.update()/per-key writes) - see
        # module docstring. Groups first, then enabled-cameras (the same
        # order they were validated in) - a reader that runs between the two
        # assignments only ever sees a MORE restrictive transient state
        # (e.g. a camera newly added to a group but not yet in the new
        # enabled-set falls through to "unauthorized"; a camera dropped from
        # a group but not yet re-checked against the new enabled-set falls
        # through to "authorized but ungrouped", which resolves to an empty,
        # not unrestricted, candidate set) - never a more permissive one.
        config.REID_CAMERA_GROUPS = normalized_groups
        config.REID_ENABLED_CAMERAS = normalized_enabled

        self.refreshes += 1
        self.last_success = time.time()

        # Always announce the startup fetch (it is the static -> dashboard
        # handover the whole feature exists for, worth logging even if the
        # dashboard's config happens to equal the static default); a later
        # refresh only logs when something actually changed, so a healthy
        # long-running process does not print this block every 30 seconds.
        if changed or context == "startup":
            _log(f"loaded {len(normalized_groups)} group(s) and "
                 f"{len(normalized_enabled)} Re-ID-authorized camera(s) from dashboard ({context})")
            for name, cfg in normalized_groups.items():
                _describe_group(name, cfg)
            _log("configuration refreshed")
