"""
Production Re-ID adapter (Phase 1) - a thin, fail-safe wrapper around the
validated reid_poc/ library, for multicam_inf.py to call.

    TrackedObject (tracking.py, UNMODIFIED)
        |
        v
    ReIDAdapter.process_tracks()   <- per camera, per tick (cheap, queues crops)
        |
    ReIDAdapter.flush()            <- once per tick, across every camera
        |
        v
    OSNet-AIN (batched)  ->  GlobalReIDManager  ->  Store (memory | Qdrant)

DO NOT replace JeztSort. This module NEVER reads or writes tracking.py's
track_id, never touches TrackManager, and produces a SEPARATE
global_person_id alongside the local track_id - exactly the relationship
the whole reid_poc/ effort has maintained throughout:

    camera_id=CAM-R09, local_track_id=123  ->  global_person_id=P-0017

======================================================================
WHAT ARMS THIS ADAPTER: the dashboard, and only the dashboard
======================================================================
Re-ID runs for a camera when the DASHBOARD authorises it - Camera.reid_enabled,
carried into config.REID_ENABLED_CAMERAS by reid_config_provider.py. There is
no environment switch that can turn Re-ID on or off, because a second switch
could disagree with the dashboard; PPE lost three hours of a production test to
exactly that failure mode, and the same gate is not kept here.

    Dashboard (Camera.reid_enabled / ReIDCameraGroup)
        -> reid_config_provider
        -> config.REID_ENABLED_CAMERAS  +  config.REID_CAMERA_GROUPS
        -> this adapter, and reid_manager.needs_observation()

When the dashboard authorises NO camera, this module does not import
torch/torchreid/qdrant-client, does not construct OSNetReID, does not construct
GlobalReIDManager, does not construct a Store, and does not import anything
from reid_poc/ AT ALL - see _init_reid_poc(), only ever called from inside
`if self.enabled:`. There is no model/store object in existence to have
overhead, not just a fast early return. process_tracks()/flush() are single
boolean-check no-ops. That property is preserved exactly; what changed is that
the DASHBOARD decides it rather than an environment variable.

The authorisation set is read at construction through one synchronous,
bounded-timeout dashboard fetch (ReIDConfigProvider's own startup fetch, which
imports no torch), so the decision is made from live configuration rather than
from the static fallback.

Mode: Phase 1 implements "observe" only - Re-ID decisions are printed/logged,
and NOTHING about existing production behaviour changes: no event schema
change, no dashboard field, no tracker write-back. An "active"/association mode
is a later phase.

======================================================================
Error isolation (hard requirement)
======================================================================
Every method that touches reid_poc/OSNet/Qdrant is wrapped so a Re-ID
failure can NEVER propagate into the caller - multicam_inf.py's RTSP
ingestion, YOLO, JeztSort, zones, events and evidence must be completely
unaffected by any Re-ID exception, OSNet error, malformed crop, or Qdrant
outage/timeout. A single exception during __init__ (bad reid_poc import,
model load failure, store connect failure) permanently disables this
instance for the rest of the process (self.enabled = False) rather than
raising - multicam_inf.py's own module-level
`reid_adapter_instance = ReIDAdapter()` call must never be able to crash
production startup. Errors are logged, rate-limited per (call site, error
type) pair via REID_ERROR_LOG_INTERVAL_SECONDS, never spammed.

======================================================================
Qdrant isolation
======================================================================
REID_PROD_* config below is INDEPENDENT of reid_poc/config.py's own
QDRANT_HOST/PORT/COLLECTION (which point at that PoC's own isolated test
instance) - production points at its own configured endpoint/collection.
Connecting to the KNOWN SHARED, multi-tenant production Qdrant
(`qdrant_memory`, port 6333 - confirmed via read-only audit to serve
unrelated products: corporate_memory, user_memory_*, video_faces,
video_objects, etc., NONE of which are CCTV-related) is refused outright
unless REID_PROD_ALLOW_SHARED_QDRANT is explicitly set - see
_build_store(). This adapter must never create collections on, or send
traffic to, that shared instance.
"""
import os
import sys
import time


def _bool_env(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ============================================================
# Phase 1 mode. Not a feature flag: the dashboard decides whether Re-ID runs
# (see the module docstring). "observe" is the only behaviour Phase 1
# implements, so it is a constant rather than a switch that could be set to a
# mode with no implementation behind it.
# ============================================================

REID_PRODUCTION_MODE = "observe"


def _dashboard_authorised_cameras():
    """
    The cameras the DASHBOARD has authorised for Re-ID, read before any heavy
    import happens.

    Constructing ReIDConfigProvider performs its own synchronous, bounded-
    timeout startup fetch and does NOT start its background thread (that is
    start(), called later from _init_reid_poc) - so this costs one short HTTP
    request and imports no torch. A failure leaves reid_poc/config.py's static
    value in place, exactly as the provider already guarantees, so an
    unreachable dashboard degrades to the last known configuration rather than
    silently disabling Re-ID.

    Returns an empty frozenset if reid_poc is not importable at all, which
    correctly reads as "Re-ID cannot run here".
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    reid_poc_dir = os.path.join(this_dir, "reid_poc")

    for path in (this_dir, reid_poc_dir):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    try:
        import config as reid_poc_config
        import reid_config_provider

        reid_config_provider.ReIDConfigProvider()
        return frozenset(reid_poc_config.REID_ENABLED_CAMERAS or ())
    except Exception as exc:                          # noqa: BLE001
        print(f"[REID-ADAPTER] could not read the dashboard Re-ID "
              f"configuration ({type(exc).__name__}: {exc}) - Re-ID off")
        return frozenset()

# ============================================================
# Production Re-ID store - deliberately separate names from reid_poc's own
# REID_STORE/QDRANT_HOST/QDRANT_PORT/QDRANT_COLLECTION.
# ============================================================

REID_PROD_STORE = os.getenv("REID_PROD_STORE", "memory").strip().lower()

# Defaults to the SAME isolated PoC Qdrant instance (127.0.0.1:6343) the
# reid_poc/ test suites already use - safe as a Phase-1 default because that
# instance is definitionally never the shared production one (6333), not
# because it's the intended long-term production endpoint. Point this at
# real, separate production Qdrant infrastructure once Phase 3 (production
# Qdrant collection design) is decided - see the Phase 1 report.
REID_PROD_QDRANT_HOST = os.getenv("REID_PROD_QDRANT_HOST", "127.0.0.1")
REID_PROD_QDRANT_PORT = int(os.getenv("REID_PROD_QDRANT_PORT", "6343"))
REID_PROD_QDRANT_COLLECTION = os.getenv("REID_PROD_QDRANT_COLLECTION", "cctv_person_reid_production")
REID_PROD_QDRANT_TIMEOUT_SECONDS = float(os.getenv("REID_PROD_QDRANT_TIMEOUT_SECONDS", "2.0"))

# Hard safety rail: refuse the known shared, multi-tenant production Qdrant
# port outright unless explicitly overridden. See module docstring.
_BLOCKED_QDRANT_PORTS = {6333}
REID_PROD_ALLOW_SHARED_QDRANT = _bool_env("REID_PROD_ALLOW_SHARED_QDRANT", False)

# PRODUCTION-HARDENING P1 FIX: defer Qdrant upsert()'s network round-trip off
# the inference thread (see reid_manager.AsyncUpsertStore - search() stays
# fully synchronous/unchanged; only upsert() is queued to a background
# writer). Defaults to True - THIS IS THE FIX being delivered here, not an
# opt-in experiment - matching how every other Phase-1/2A/hardening fix in
# this file has shipped as the new default behaviour rather than a flag
# someone has to remember to turn on. Only has any effect when
# REID_PROD_STORE=="qdrant" (see _build_store()) - the memory store's
# upsert() is already effectively instant, so wrapping it would only add a
# queue/thread for no benefit.
REID_PROD_ASYNC_UPSERT = _bool_env("REID_PROD_ASYNC_UPSERT", True)

REID_ERROR_LOG_INTERVAL_SECONDS = float(os.getenv("REID_ERROR_LOG_INTERVAL_SECONDS", "30.0"))
REID_STATS_INTERVAL_SECONDS = float(os.getenv("REID_STATS_INTERVAL_SECONDS", "30.0"))

# Deliberately its OWN, independently-defaulted setting - NOT inherited from
# reid_poc/config.py's REID_CAMERA_MODE. reid_poc/config.py is being actively
# experimented on (same_camera / same_and_cross modes, camera-group tuning)
# by the PoC work this adapter wraps; production must not silently change
# behaviour because someone changed that experiment's config. Phase 1's
# intended default is the conservative, already-validated cross-camera-only
# policy regardless of whatever mode the PoC is currently testing.
# REID_CAMERA_GROUPS/REID_SIMILARITY_THRESHOLD/REID_MIN_MATCH_MARGIN are
# still reused directly from reid_poc/config.py (see module docstring/
# CAMERA GROUP SUPPORT in the Phase 1 report) - those are shared numeric/
# structural config, not an experimental-mode toggle.
REID_PROD_CAMERA_MODE = os.getenv("REID_PROD_CAMERA_MODE", "cross_camera_only").strip().lower()
if REID_PROD_CAMERA_MODE not in ("cross_camera_only", "same_camera", "same_and_cross"):
    raise ValueError(
        f"REID_PROD_CAMERA_MODE={REID_PROD_CAMERA_MODE!r} is not valid - "
        f"must be one of cross_camera_only/same_camera/same_and_cross"
    )


# ============================================================
# CANONICAL DECISION COUNTERS
#
# reid_manager.py emits ten different statuses across three camera modes.
# These four names are the operational view: what happened to an identity,
# independent of which mode produced it, so a before/after comparison across
# a mode change compares like with like.
#
#   REID_MATCH                 an existing identity was reused
#   REID_NEW                   a fresh identity was created
#   REID_SKIPPED_SAME_CAMERA   a same-camera candidate was found and NOT used
#   REID_SKIPPED_CROSS_CAMERA  the cross-camera pool was excluded by policy
#
# THE TWO "SKIPPED" COUNTERS ARE NOT SYMMETRIC, and must not be read as if
# they were:
#   SKIPPED_SAME_CAMERA is EVIDENCE. The manager only reports it after a
#     second search confirmed a same-camera candidate at or above the
#     camera's threshold. Every count is a real recovery that was refused.
#   SKIPPED_CROSS_CAMERA is POLICY. It counts bootstrap decisions on a
#     camera resolved to "same_camera", where the cross-camera pool was
#     never searched at all. It does NOT claim a cross-camera candidate
#     existed - nobody looked. Its purpose is to prove the camera-local
#     restriction is actually in force on the cameras that are supposed to
#     have it, which is exactly the property an ungrouped camera must hold.
# ============================================================

REID_MATCH = "REID_MATCH"
REID_NEW = "REID_NEW"
REID_SKIPPED_SAME_CAMERA = "REID_SKIPPED_SAME_CAMERA"
REID_SKIPPED_CROSS_CAMERA = "REID_SKIPPED_CROSS_CAMERA"

#: manager status -> canonical counter. A status absent from this map (
#: UNCERTAIN, SAME_CAM_UNCERTAIN, SEARCH_FAILED, REFRESH) resolved no
#: identity and is deliberately counted under none of the four - they are
#: still visible individually in the raw decisions= breakdown.
_CANONICAL_STATUS = {
    "MATCH": REID_MATCH,
    "SAME_CAM_MATCH": REID_MATCH,
    "CROSS_CAM_MATCH": REID_MATCH,
    "NEW": REID_NEW,
    "SKIPPED_SAME_CAMERA": REID_SKIPPED_SAME_CAMERA,
    "SAME_CAM_BLOCKED": REID_SKIPPED_SAME_CAMERA,
}

#: Statuses produced by a bootstrap decision (as opposed to REFRESH, which
#: only updates an already-resolved identity's gallery).
_DECISION_STATUSES = frozenset({
    "NEW", "MATCH", "UNCERTAIN", "SKIPPED_SAME_CAMERA", "SEARCH_FAILED",
    "SAME_CAM_MATCH", "CROSS_CAM_MATCH", "SAME_CAM_UNCERTAIN", "SAME_CAM_BLOCKED",
})

#: Statuses that only arise from the same-camera candidate pool - the ones
#: worth a detailed per-decision line while same-camera recovery is being
#: validated.
_SAME_CAMERA_STATUSES = frozenset({
    "SAME_CAM_MATCH", "SAME_CAM_UNCERTAIN", "SAME_CAM_BLOCKED", "SKIPPED_SAME_CAMERA",
})


class _Stats:
    """Re-ID-only counters - deliberately separate from multicam_inf.py's own
    camera_stats, per the requirement that Re-ID metrics be independent of
    existing pipeline metrics."""

    __slots__ = ("eligible", "crops", "osnet_calls", "osnet_ms", "decisions",
                 "errors", "canonical", "recoveries")

    def __init__(self):
        self.eligible = 0
        self.crops = 0
        self.osnet_calls = 0
        self.osnet_ms = 0.0
        self.decisions = {}
        self.errors = 0
        # Cumulative across the whole process, NOT reset each stats interval
        # (unlike eligible/crops/osnet_*) - these are the totals a before/
        # after comparison is read from, so a run's figure must not depend on
        # which 30-second window it was sampled in.
        self.canonical = {
            REID_MATCH: 0, REID_NEW: 0,
            REID_SKIPPED_SAME_CAMERA: 0, REID_SKIPPED_CROSS_CAMERA: 0,
        }
        #: same-camera MATCHes where the identity was previously held by a
        #: DIFFERENT local track on this camera - i.e. a fragmented identity
        #: actually rejoined. The headline number this whole change exists
        #: to produce.
        self.recoveries = 0

    def record_osnet(self, n, ms):
        self.crops += n
        self.osnet_calls += 1
        self.osnet_ms += ms

    def record_decision(self, status, camera_mode=None):
        self.decisions[status] = self.decisions.get(status, 0) + 1

        canonical = _CANONICAL_STATUS.get(status)
        if canonical is not None:
            self.canonical[canonical] += 1

        # Policy counter, not evidence - see the block comment above. Counted
        # per bootstrap DECISION, so it is directly comparable with the other
        # three rather than being inflated by REFRESH ticks.
        if camera_mode == "same_camera" and status in _DECISION_STATUSES:
            self.canonical[REID_SKIPPED_CROSS_CAMERA] += 1


class ReIDAdapter:
    """
    Production-facing Re-ID entry point. multicam_inf.py only ever calls
    process_tracks() / flush() / close() - OSNet, Qdrant, GlobalReIDManager,
    cosine similarity and gallery management all stay behind this boundary.

        adapter = ReIDAdapter()                                  # once, at startup
        ...
        adapter.process_tracks(camera_id, tracked_objects, frame, ts)   # per camera, per tick
        ...
        adapter.flush()                                           # once per tick, after every camera
        ...
        adapter.close()                                           # at shutdown
    """

    def enabled_cameras(self):
        """
        The cameras the dashboard has authorised for Re-ID, for the startup
        status block. Re-read live so it reflects the provider's latest
        refresh rather than the value captured at construction.
        """
        if self._reid_poc_config is not None:
            return sorted(getattr(self._reid_poc_config,
                                  "REID_ENABLED_CAMERAS", ()) or ())
        return sorted(self._authorised)

    def __init__(self):
        self._authorised = _dashboard_authorised_cameras()
        self.enabled = bool(self._authorised)
        self.mode = REID_PRODUCTION_MODE
        self._manager = None
        self._osnet = None
        self._crop_fn = None
        self._person_group = None
        self._store = None
        self._async_store = None
        self._config_provider = None
        self._observability = None
        self._pending = []
        self._last_error_log = {}
        self._stats = _Stats()
        self._last_stats_print = time.monotonic()
        self._last_track_cleanup = time.monotonic()
        self._reid_poc_config = None

        if not self.enabled:
            print("[REID-ADAPTER] no camera is Re-ID authorised in the dashboard "
                  "- no Re-ID model/store initialised")
            return

        try:
            self._init_reid_poc()
            print(f"[REID-ADAPTER] enabled  mode={self.mode}  store={REID_PROD_STORE}")
        except Exception as exc:  # noqa: BLE001 - must never propagate from __init__
            print(f"[REID-ADAPTER] initialisation FAILED - Re-ID disabled for this process: "
                  f"{type(exc).__name__}: {exc}")
            self.enabled = False
            self._manager = None
            self._osnet = None
            self._store = None
            self._async_store = None
            self._config_provider = None
            self._observability = None
            self._reid_poc_config = None

    # ------------------------------------------------------------- lifecycle
    def _init_reid_poc(self):
        """Only ever called when enabled - reid_poc/torch/torchreid/
        qdrant-client are not imported at all otherwise. Mirrors
        reid_poc/test_reid_tracking.py's own sys.path fix exactly (THIS_DIR
        must be inserted LAST so `import config`/`import reid_manager` etc.
        resolve to reid_poc/'s own modules, not some other same-named module
        elsewhere on the path)."""
        this_dir = os.path.dirname(os.path.abspath(__file__))
        reid_poc_dir = os.path.join(this_dir, "reid_poc")
        for path in (this_dir, reid_poc_dir):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)

        import reid as reid_module   # reid_poc/reid.py - OSNet-AIN wrapper
        import reid_manager          # reid_poc/reid_manager.py
        import tracking as tracking_module   # production, UNMODIFIED - for the PERSON_GROUP constant only
        import reid_config_provider  # AI_inferencing/reid_config_provider.py (Phase 2A)
        import config as reid_poc_config     # reid_poc/config.py - REID_TRACK_CLEANUP_INTERVAL_SECONDS
                                              # only (a structural/numeric value reused directly, exactly
                                              # like REID_CAMERA_GROUPS already is - see module docstring's
                                              # "shared numeric/structural config" vs. "independently-
                                              # defaulted mode toggle" distinction); NOT a second,
                                              # independent production copy of this setting.

        # PHASE 3A OBSERVABILITY - a tiny, dependency-free, opt-in module
        # living alongside this file (not reid_poc/) - not imported
        # conditionally the way torch/torchreid/qdrant-client are, since it
        # has no heavy dependencies of its own to avoid; disabled by default
        # (REID_OBSERVABILITY_ENABLED) so constructing it here costs nothing
        # beyond a few attribute assignments unless explicitly turned on.
        import reid_observability
        self._observability = reid_observability.ReIDObservabilityLogger()

        self._reid_poc_config = reid_poc_config
        self._crop_fn = reid_module.crop_person
        self._person_group = tracking_module.PERSON_GROUP
        self._osnet = reid_module.OSNetReID()

        raw_store = self._build_store(reid_manager)
        first_identity_seq = self._first_identity_seq(raw_store)

        # PRODUCTION-HARDENING P1 FIX: only the Qdrant-backed store has an
        # expensive synchronous upsert() worth deferring (InMemoryReIDStore's
        # upsert() is already a plain dict append) - see reid_manager.
        # AsyncUpsertStore's own docstring for the full design/verification.
        # Layered UNDER TimingStore deliberately: TimingStore.upsert_ms then
        # measures what the inference thread actually pays (the queue put),
        # which is the metric this fix is supposed to shrink - AsyncUpsertStore's
        # own written/dropped_*/queue_depth() counters (read in _maybe_print_
        # stats() below) describe the writer thread's real, separate behaviour.
        if REID_PROD_STORE == "qdrant" and REID_PROD_ASYNC_UPSERT:
            self._async_store = reid_manager.AsyncUpsertStore(raw_store)
            raw_store = self._async_store

        self._store = reid_manager.TimingStore(raw_store)   # reused, not reinvented - gives search_ms/upsert_ms for stats
        self._manager = reid_manager.GlobalReIDManager(self._store, camera_mode=REID_PROD_CAMERA_MODE,
                                                       first_identity_seq=first_identity_seq)

        # Phase 2A: dashboard-backed camera-group config (threshold/margin/
        # camera_mode/topology). The provider's constructor already performs
        # ONE synchronous startup fetch (static config -> dashboard fetch ->
        # valid config replaces static, per its own docstring) before this
        # line returns; start() then launches the periodic background
        # refresh thread. A dashboard-fetch failure here never raises - see
        # reid_config_provider.ReIDConfigProvider._fetch_and_apply() - so a
        # dashboard outage at startup never disables Re-ID for this process,
        # it just means the static config stays live until the dashboard
        # becomes reachable.
        self._config_provider = reid_config_provider.ReIDConfigProvider()
        self._config_provider.start()

        self._print_resolved_modes()

    def _print_resolved_modes(self):
        """
        One line per authorised camera, at startup, naming the camera_mode it
        actually resolved to.

        A rollout that silently fails to apply is the failure mode worth
        spending five lines of log on: "CAM-R16 is supposed to be doing
        same-camera recovery now" is otherwise only checkable by inferring it
        from decision statuses much later. This states it directly, before
        any frame is processed.
        """
        try:
            rollout = sorted(self._reid_poc_config.REID_SAME_CAMERA_CAMERAS or ())
            print(f"[REID-MODE] same-camera rollout set: "
                  f"{', '.join(rollout) if rollout else '(empty - no camera opted in)'}")

            for camera_id in self.enabled_cameras():
                policy = self._manager._policy_for(camera_id)
                print(f"[REID-MODE] camera={camera_id} "
                      f"group={policy['group'] or '-'} "
                      f"mode={policy['camera_mode']} "
                      f"threshold={policy['similarity_threshold']:.2f} "
                      f"margin={policy['min_match_margin']:.2f}")
        except Exception as exc:  # noqa: BLE001
            self._log_error("print_resolved_modes", exc)

    @staticmethod
    def _build_store(reid_manager):
        if REID_PROD_STORE != "qdrant":
            return reid_manager.InMemoryReIDStore()

        if REID_PROD_QDRANT_PORT in _BLOCKED_QDRANT_PORTS and not REID_PROD_ALLOW_SHARED_QDRANT:
            raise RuntimeError(
                f"REID_PROD_QDRANT_PORT={REID_PROD_QDRANT_PORT} is the KNOWN SHARED, multi-tenant "
                f"production Qdrant - refusing to connect the Re-ID adapter to it without "
                f"REID_PROD_ALLOW_SHARED_QDRANT=1. Use an isolated Re-ID Qdrant instance instead."
            )

        import qdrant_reid  # reid_poc/qdrant_reid.py

        return qdrant_reid.QdrantReIDStore(
            collection_name=REID_PROD_QDRANT_COLLECTION,
            host=REID_PROD_QDRANT_HOST,
            port=REID_PROD_QDRANT_PORT,
            timeout=REID_PROD_QDRANT_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _first_identity_seq(store):
        """
        Where this process's P-#### numbering starts.

        The memory store is empty at every start, so P-0001. The Qdrant store
        persists across restarts, so numbering continues after the highest id
        already in the collection - one read, here, before any identity can
        be created. If that read fails this RAISES: __init__ then disables
        Re-ID for the process rather than restart at P-0001 and hand stored
        identities' ids to new people.
        """
        if REID_PROD_STORE != "qdrant":
            return 1

        first = store.highest_identity_seq() + 1
        print(f"[REID-ADAPTER] persistent identity numbering: "
              f"collection={store.collection_name} next=P-{first:04d}")
        return first

    def close(self):
        """
        PRODUCTION-HARDENING P1 FIX shutdown ordering (each step wrapped in
        its own try/except - one failing must not skip the rest, exactly
        like this method's pre-existing steps already do):

          1. stop accepting new persistence work / 2. stop inference-config
             activity - satisfied by the EXISTING calling convention, not
             new code here: multicam_inf.py only calls close() AFTER it has
             already stopped calling process_tracks()/flush() for good (see
             the class docstring's usage example), so no new upsert() call
             is expected to arrive during shutdown. self._config_provider.
             stop() (pre-existing) additionally stops the ONE OTHER
             background thread this adapter owns, so no config refresh is
             still in flight either.
          3. drain/flush the bounded async-upsert queue + 4. close that
             queue's worker thread - ONE call, self._async_store.stop():
             internally signals its writer thread to stop accepting new
             *waiting* (queued work already present is still drained) and
             joins it bounded by REID_ASYNC_UPSERT_SHUTDOWN_FLUSH_SECONDS -
             see reid_manager.AsyncUpsertStore.stop()'s own docstring. Only
             present when REID_PROD_STORE=="qdrant" and REID_PROD_ASYNC_
             UPSERT (the default) - a no-op step otherwise (self._async_store
             stays None). Logged here, not inside reid_manager.py (that
             module has no I/O of its own - see AsyncUpsertStore's docstring).
          5. close manager/store - no explicit close exists or is needed for
             GlobalReIDManager/InMemoryReIDStore/QdrantReIDStore in this
             codebase (pure in-memory data structures plus an HTTP-backed
             client with no explicit lifecycle previously called anywhere) -
             unchanged from before this fix.
          6. complete shutdown - this method returning.

        Idempotent: calling close() more than once must never raise -
        self._async_store.stop() itself is idempotent (see its own
        docstring), and every step here is already guarded by
        `if X is not None`/try-except, matching this method's pre-existing
        idempotency (see test_15_shutdown, which already calls close() twice).

        PRODUCTION-HARDENING J-1 FIX: each step below is now ALSO guarded by
        its own `except KeyboardInterrupt:`, narrowly scoped around that one
        operation - not one broad handler wrapping this whole method (see
        _log_shutdown_interrupt()'s own docstring for why the narrow,
        per-step shape was chosen). CONFIRMED finding: KeyboardInterrupt does
        NOT inherit from Exception, so a SIGINT landing while
        self._async_store.stop() is blocked inside its bounded
        thread.join(timeout=...) - a real, multi-second window Fix #3
        introduced that did not exist before it - used to propagate straight
        through the `except Exception` guards below, out of close() entirely,
        skipping every later step here AND (by multicam_inf.py's own
        sequencing - reid_adapter_instance.close() runs before
        ppe_adapter_instance.close()/behaviour_adapter_instance.close()/CUDA
        teardown, all in the same try block) everything after it too.
        Reproduced directly in the validation phase (Finding J-1) against a
        real KeyboardInterrupt landing mid-join(); this is the fix.

        The existing SIGTERM path (multicam_inf.py's own _handle_sigterm,
        which turns a SIGTERM into this exact KeyboardInterrupt/finally
        sequence and already guards against a REPEATED SIGTERM re-entering
        it) is unaffected - this fix only changes what happens to a
        KeyboardInterrupt ONCE INSIDE close() itself; it does not touch, and
        does not need to touch, the signal-handler registration at all (no
        new signal-handling framework introduced - see module docstring).
        """
        if not self.enabled:
            return
        try:
            if self._config_provider is not None:
                self._config_provider.stop()
        except KeyboardInterrupt:
            self._log_shutdown_interrupt("config watcher shutdown")
        except Exception as exc:  # noqa: BLE001
            self._log_error("close", exc)
        try:
            if self._async_store is not None:
                remaining = self._async_store.stop()
                if remaining:
                    print(f"[REID-ASYNC-UPSERT] shutdown: {remaining} queued upsert(s) "
                          f"not flushed in time and were discarded")
        except KeyboardInterrupt:
            self._log_shutdown_interrupt("async writer shutdown")
        except Exception as exc:  # noqa: BLE001
            self._log_error("close", exc)
        try:
            if self._osnet is not None:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except KeyboardInterrupt:
            self._log_shutdown_interrupt("CUDA teardown")
        except Exception as exc:  # noqa: BLE001
            self._log_error("close", exc)
        try:
            if self._observability is not None:
                # PHASE 3B WORKSTREAM D: these 4 counters were already
                # computed by every record() call all along - just never
                # read by anything, so an operator had no way to tell "N
                # records were dropped to sampling" without opening the
                # JSONL and counting lines by hand. One summary line,
                # printed once at shutdown - not a new per-decision log.
                obs = self._observability
                print(f"[REID-OBSERVABILITY] written={obs.records_written} "
                      f"dropped_sampled={obs.records_dropped_sampled} "
                      f"skipped_log_level={obs.records_skipped_log_level} "
                      f"errors={obs.errors}")
                self._observability.close()
        except KeyboardInterrupt:
            self._log_shutdown_interrupt("observability shutdown")
        except Exception as exc:  # noqa: BLE001
            self._log_error("close", exc)

    # ------------------------------------------------------- stable identity
    def stable_id_for(self, camera_id, track_id):
        """
        The stable application identity for one local track, or None.

        This is the ONLY thing the event pipeline calls, and it is the whole
        public surface of the "layer 3" identity: events.py holds a reference
        to this bound method and knows nothing else about Re-ID.

        None - meaning "omit metadata['stable_id']" - is the correct, normal
        answer in every one of these cases, and none of them is an error:
        Re-ID disabled, camera not authorised, a vehicle (Re-ID is
        person-only), a track still bootstrapping, or a track whose last
        decision was UNCERTAIN. Callers must never treat None as a failure.

        Never raises and never blocks: a Re-ID fault must not stop an event
        being published. On any exception the event is simply published
        without a stable_id, exactly as it is today.
        """
        if not self.enabled or self._manager is None:
            return None

        try:
            return self._manager.stable_id_for(camera_id, track_id)
        except Exception as exc:  # noqa: BLE001
            self._log_error("stable_id_for", exc)
            return None

    # -------------------------------------------------------------- process
    def process_tracks(self, camera_id, tracks, frame, timestamp):
        """
        Call once per camera, per tick, right after track_manager.update().
        Cheap: only queues eligible person crops for the batched OSNet call
        in flush() - never calls OSNet itself, never touches the store.
        Always safe to call unconditionally, enabled or not.
        """
        if not self.enabled or self._manager is None:
            return

        # Per-track try/except, not one try/except around the whole loop - a
        # single malformed track (bad attribute, unexpected type) must not
        # shadow every OTHER track queued this same tick on this camera.
        for tracked in tracks:
            try:
                if tracked.group != self._person_group:
                    continue

                self._stats.eligible += 1

                if not self._manager.needs_observation(camera_id, tracked.track_id):
                    continue

                crop = self._crop_fn(frame, tracked.bbox)
                if crop is None:
                    continue

                self._pending.append((camera_id, tracked.track_id, tracked.confidence, timestamp, crop))
            except Exception as exc:  # noqa: BLE001
                self._log_error("process_tracks", exc)

    def flush(self):
        """
        Call once per tick, after every camera's process_tracks() this tick -
        NOT inside the per-camera loop. Runs ONE batched OSNet call across
        every crop queued by every camera this tick (mirrors reid_poc's
        tested per-tick batching exactly - never one OSNet call per camera,
        never one per track), then one observe() per result. Returns the
        list of ObserveResult produced this tick (Phase 1 callers are not
        required to use it - observe mode logs internally).
        """
        if not self.enabled or self._manager is None:
            return []

        self._maybe_evict_stale_tracks()

        if not self._pending:
            self._maybe_print_stats()
            return []

        batch = self._pending
        self._pending = []

        try:
            crops = [item[4] for item in batch]
            start = time.perf_counter()
            embeddings = self._osnet.extract_batch(crops)
            osnet_ms = (time.perf_counter() - start) * 1000.0
            self._stats.record_osnet(len(crops), osnet_ms)
        except Exception as exc:  # noqa: BLE001
            self._log_error("osnet_extract_batch", exc)
            return []

        results = []
        for (camera_id, track_id, confidence, ts, _crop), embedding in zip(batch, embeddings):
            try:
                result = self._manager.observe(camera_id, track_id, embedding, confidence, timestamp=ts)
            except Exception as exc:  # noqa: BLE001
                self._log_error("observe", exc)
                continue

            if result is None:
                continue

            self._stats.record_decision(result.status, result.effective_camera_mode)

            # Same-camera decisions get a detailed, machine-parsable line -
            # this is the measurement surface for the rollout, so it is
            # unconditional rather than gated on self.mode (which only
            # controls the human-readable [REID-OBSERVE] line below).
            if result.status in _SAME_CAMERA_STATUSES:
                self._print_same_camera(camera_id, track_id, result)

            # PRODUCTION-HARDENING G-1 FIX: unconditional, NOT gated on
            # self.mode - this is an operational/infrastructure signal (a
            # Qdrant search actually failed and was retried), the same
            # category as _log_error()'s own always-on logging, not a
            # per-decision display line like _print_observe() below. Only
            # prints anything when result.search_retry_events is populated,
            # which only happens on an actual retry - "no repeated search
            # retry logs during healthy operation" holds by construction.
            if result.search_retry_events:
                self._print_search_retries(camera_id, track_id, result.search_retry_events)

            # PHASE 3A OBSERVABILITY: unconditional call, NOT gated on
            # self.mode - record() itself is the on/off switch
            # (REID_OBSERVABILITY_ENABLED), a single boolean check when off.
            # Read-only with respect to `result` - cannot affect the decision
            # that already happened.
            if self._observability is not None:
                self._observability.record(camera_id, track_id, result)

            if self.mode == "observe":
                self._print_observe(camera_id, track_id, result)

            results.append(result)

        self._maybe_print_stats()
        return results

    # -------------------------------------------------------------- logging
    def _print_same_camera(self, camera_id, track_id, result):
        """
        One fixed-field line per same-camera decision - the measurement
        surface for the rollout, designed to be grepped and counted rather
        than read.

            [REID-SAMECAM] camera=CAM-R16 track=P-O4E2-0031 stable=P-0007
                           prev_track=P-O4E2-0017 prev_stable=- gap=4.21
                           sim=0.87 second=0.61 margin=0.26 thr=0.75
                           group=DISTRI WAIT BUILDING mode=same_and_cross
                           decision=SAME_CAM_MATCH recovered=yes

        recovered=yes is the claim this whole change has to earn: the
        identity assigned to THIS local track was already held by a DIFFERENT
        local track on THIS camera. That is one fragmented identity rejoined,
        and it is counted rather than inferred. prev_track=- with
        recovered=no is a genuinely new identity - or a recovery whose
        predecessor has already aged out of the manager's track map, which is
        reported as unknown rather than guessed at.

        For a SKIPPED_* decision, sim= deliberately reports the SKIPPED
        candidate's own similarity (from result.skipped_candidate), not
        result.similarity - result.similarity there is the best score from
        the pool that was actually searched, which for a suppressed
        same-camera match is usually the cross-camera pool and usually
        absent. Logging the wrong one of those two is what made the existing
        [REID-OBSERVE] line unable to size this problem at all.
        """
        prev_track, gap = None, None
        try:
            prev_track, gap = self._manager.recovery_provenance(
                camera_id, result.global_id, track_id
            )
        except Exception as exc:  # noqa: BLE001
            self._log_error("recovery_provenance", exc)

        recovered = prev_track is not None and result.status in (
            "SAME_CAM_MATCH", "MATCH", "CROSS_CAM_MATCH",
        )
        if recovered:
            self._stats.recoveries += 1

        similarity = result.similarity
        skipped_id = None
        if result.skipped_candidate is not None:
            skipped_id, similarity = result.skipped_candidate

        def fmt(value, spec=".2f"):
            return "-" if value is None else format(value, spec)

        print(
            f"[REID-SAMECAM] camera={camera_id} track={track_id} "
            f"stable={result.global_id or skipped_id or '-'} "
            f"prev_track={prev_track or '-'} "
            f"prev_stable={skipped_id or '-'} "
            f"gap={fmt(gap)} sim={fmt(similarity)} "
            f"second={fmt(result.second_similarity)} margin={fmt(result.margin)} "
            f"thr={fmt(result.effective_threshold)} "
            f"group={result.group or '-'} "
            f"mode={result.effective_camera_mode or '-'} "
            f"decision={result.status} "
            f"recovered={'yes' if recovered else 'no'}"
        )

    @staticmethod
    def _print_search_retries(camera_id, track_id, retry_events):
        """
        PRODUCTION-HARDENING G-1 FIX. retry_events (see reid_manager.
        ObserveResult.search_retry_events's own docstring) is a small,
        bounded list (at most REID_SEARCH_RETRY_MAX_ATTEMPTS entries) built
        entirely INSIDE the one observe() call that just returned - printed
        here, right after, rather than truly interleaved in real time during
        the retries themselves (reid_manager.py has no I/O of its own by
        design - see AsyncUpsertStore's docstring for the same convention).
        The few-millisecond difference between "as it happened" and "printed
        immediately after observe() returns" carries no operational meaning.
        """
        for event in retry_events:
            if event["outcome"] == "retry":
                print(f"[REID-SEARCH-RETRY] camera={camera_id} track={track_id} "
                      f"attempt={event['attempt']}/{event['max_attempts']}")
            else:
                print(f"[REID-SEARCH-FAILED] camera={camera_id} track={track_id} "
                      f"attempts={event['max_attempts']}")

    @staticmethod
    def _print_observe(camera_id, track_id, result):
        sim = "-" if result.similarity is None else f"{result.similarity:.2f}"
        second = "-" if result.second_similarity is None else f"{result.second_similarity:.2f}"
        margin = "-" if result.margin is None else f"{result.margin:.2f}"
        global_id = result.global_id or "-"
        group_part = f" group={result.group}" if result.group else ""
        print(f"[REID-OBSERVE] camera={camera_id} track={track_id} global={global_id} "
              f"similarity={sim} second={second} margin={margin} status={result.status}{group_part}")

    def _log_error(self, where, exc):
        self._stats.errors += 1
        key = (where, type(exc).__name__)
        now = time.monotonic()
        last = self._last_error_log.get(key)
        if last is not None and (now - last) < REID_ERROR_LOG_INTERVAL_SECONDS:
            return
        self._last_error_log[key] = now
        print(f"[REID-ERROR] {where}: {type(exc).__name__}: {exc}")

    @staticmethod
    def _log_shutdown_interrupt(where):
        """
        PRODUCTION-HARDENING J-1 FIX. Deliberately a SEPARATE, plain print -
        not routed through _log_error() - because a Ctrl+C during shutdown
        is an EXPECTED user action, not an error: it must never increment
        self._stats.errors, never be rate-limited (close() runs at most a
        handful of times per process, never in a hot loop, so log-spam
        protection has nothing to protect against here), and must never
        print a traceback the way an uncaught KeyboardInterrupt otherwise
        would - one concise line, then the caller's except block lets
        close() continue to its next step.
        """
        print(f"[REID-SHUTDOWN] {where} interrupted; continuing cleanup")

    def _maybe_evict_stale_tracks(self):
        """
        PRODUCTION-HARDENING P1 FIX cadence gate - bounds GlobalReIDManager.
        _tracks (see reid_manager.GlobalReIDManager.evict_stale_tracks()'s own
        extensive docstring for why it is otherwise unbounded over long
        uptime - CONFIRMED finding). Mirrors _maybe_print_stats()'s own
        elapsed-time-gate pattern exactly: called unconditionally every
        flush() tick, but only actually sweeps once every
        config.REID_TRACK_CLEANUP_INTERVAL_SECONDS has elapsed - NOT a
        per-frame or per-tick scan. Called before the _pending emptiness
        check (unlike _maybe_print_stats()) so a camera that stops producing
        eligible tracks entirely still gets its stale entries swept on this
        same wall-clock cadence, not only while there happens to be OSNet
        work queued.

        evict_stale_tracks() itself pulls ttl_seconds fresh from
        config.REID_TRACK_STATE_TTL_SECONDS on every call (same "None -> read
        config.* fresh" convention GlobalReIDManager.__init__ already uses) -
        this method does not pass one explicitly, so a live dashboard-driven
        change to that setting would take effect immediately; today nothing
        refreshes it at runtime, but the manager makes no assumption either
        way.

        Wrapped in the same try/except-and-rate-limited-log pattern as every
        other reid_poc-touching call in this adapter - a bug in eviction must
        never take down the inference tick that happens to trigger it.
        """
        now = time.monotonic()
        interval = self._reid_poc_config.REID_TRACK_CLEANUP_INTERVAL_SECONDS
        if (now - self._last_track_cleanup) < interval:
            return
        self._last_track_cleanup = now

        try:
            evicted = self._manager.evict_stale_tracks()
            if evicted:
                print(f"[REID-CLEANUP] evicted {evicted} stale track(s), "
                      f"{self._manager.track_count()} remaining")
        except Exception as exc:  # noqa: BLE001
            self._log_error("evict_stale_tracks", exc)

    def _maybe_print_stats(self):
        now = time.monotonic()
        if (now - self._last_stats_print) < REID_STATS_INTERVAL_SECONDS:
            return
        self._last_stats_print = now

        search_ms = self._store.search_ms if self._store is not None else []
        upsert_ms = self._store.upsert_ms if self._store is not None else []
        avg_search = sum(search_ms) / len(search_ms) if search_ms else 0.0
        avg_upsert = sum(upsert_ms) / len(upsert_ms) if upsert_ms else 0.0
        avg_osnet = self._stats.osnet_ms / self._stats.osnet_calls if self._stats.osnet_calls else 0.0

        async_part = ""
        if self._async_store is not None:
            # PRODUCTION-HARDENING P1 FIX observability: this is the writer
            # thread's OWN view of Qdrant write behaviour, cumulative (not
            # reset each interval, unlike the counters above) - unlike
            # avg_upsert_ms above, which now only measures the queue put (see
            # AsyncUpsertStore's docstring), these describe whether the
            # background writer is actually keeping up with real Qdrant.
            # PHASE 3B WORKSTREAM B: avg_real_upsert_ms closes that exact gap
            # - the REAL network write time, timed on the writer thread
            # itself (AsyncUpsertStore.write_ms). Reset each interval (like
            # search_ms/upsert_ms below), unlike the 3 counters above,
            # because it's a list that would otherwise grow unboundedly over
            # a long-running process - the counters are plain ints, so they
            # don't have that problem and stay cumulative on purpose.
            write_ms = self._async_store.write_ms
            avg_real_upsert = sum(write_ms) / len(write_ms) if write_ms else 0.0
            async_part = (
                f" async_queue_depth={self._async_store.queue_depth()} "
                f"async_written={self._async_store.written} "
                f"async_dropped_queue_full={self._async_store.dropped_queue_full} "
                f"async_dropped_after_retries={self._async_store.dropped_after_retries} "
                f"avg_real_upsert_ms={avg_real_upsert:.2f}"
            )
            write_ms.clear()

        # Cumulative canonical counters on their own line - the four names the
        # rollout is measured on, kept separate from the per-interval
        # throughput figures below so the two are never confused for one
        # another when a run is compared against a baseline.
        canonical = self._stats.canonical
        print(f"[REID-COUNTERS] {REID_MATCH}={canonical[REID_MATCH]} "
              f"{REID_NEW}={canonical[REID_NEW]} "
              f"{REID_SKIPPED_SAME_CAMERA}={canonical[REID_SKIPPED_SAME_CAMERA]} "
              f"{REID_SKIPPED_CROSS_CAMERA}={canonical[REID_SKIPPED_CROSS_CAMERA]} "
              f"recoveries={self._stats.recoveries}")

        print(f"[REID-STATS] eligible={self._stats.eligible} crops={self._stats.crops} "
              f"osnet_calls={self._stats.osnet_calls} avg_osnet_ms={avg_osnet:.1f} "
              f"search_calls={len(search_ms)} avg_search_ms={avg_search:.2f} "
              f"upsert_calls={len(upsert_ms)} avg_upsert_ms={avg_upsert:.2f} "
              f"decisions={self._stats.decisions} errors={self._stats.errors} "
              f"identities={self._manager.identity_count() if self._manager else 0} "
              f"tracks={self._manager.track_count() if self._manager else 0}"
              f"{async_part}")

        if self._store is not None:
            search_ms.clear()
            upsert_ms.clear()
        self._stats.eligible = 0
        self._stats.crops = 0
        self._stats.osnet_calls = 0
        self._stats.osnet_ms = 0.0
