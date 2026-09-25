"""
Re-ID PoC configuration - every threshold in one place, every one overridable
by an environment variable, nothing hardcoded in the logic modules.

This file is the ONLY place that should ever need editing while tuning the
PoC. reid.py / reid_manager.py / qdrant_reid.py / test_reid_tracking.py all
read their settings from here.
"""
import json
import os


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ============================================================
# OSNet-AIN model
# ============================================================

# torchreid model name. osnet_ain_x1_0 is the AIN (instance+batch normalised)
# variant chosen for this PoC specifically for better generalisation across
# different cameras/lighting than plain osnet_x1_0 (also verified working in
# this environment, see the top-level reid.py/test_reid.py probe scripts).
REID_MODEL_NAME = os.getenv("REID_MODEL_NAME", "osnet_ain_x1_0")
REID_DEVICE = os.getenv("REID_DEVICE", "cuda")

# torchreid's FeatureExtractor output is NOT pre-normalised (measured raw L2
# norm ~31.7 on this model) - reid.py L2-normalises to unit length itself, so
# every stored/compared embedding has norm 1.0 and cosine similarity reduces
# to a plain dot product.
EMBEDDING_DIM = 512


# ============================================================
# Crop extraction (bbox -> the image OSNet actually sees)
# ============================================================

# Padding added around the JeztSort bbox before cropping, as a fraction of
# the box's own width/height on each side. OSNet's training crops (Market-
# 1501/MSMT17 style) are fairly tight around the person; a little padding
# forgives a slightly-tight detector box without diluting the crop with
# background.
#   TOO LOW  -> limbs/edges clipped by a tight YOLO box are cropped out too,
#               losing appearance information the embedding could have used.
#   TOO HIGH -> background dominates the crop, diluting the person's
#               appearance signal and making different people's crops look
#               more similar than they are.
REID_CROP_PADDING_FRAC = float(os.getenv("REID_CROP_PADDING_FRAC", "0.05"))

# Reject crops smaller than this (post-clamp-to-frame, pre-padding) - a tiny
# box rarely carries enough resolution for OSNet's 256x128 input to encode
# useful appearance detail, and mostly reflects a distant or partially
# off-frame detection.
REID_MIN_CROP_WIDTH = int(os.getenv("REID_MIN_CROP_WIDTH", "40"))
REID_MIN_CROP_HEIGHT = int(os.getenv("REID_MIN_CROP_HEIGHT", "80"))

# Reject a box whose width/height ratio is implausible for a standing or
# seated person (a badly merged detection, or two people counted as one box,
# tends to be unusually wide relative to its height).
REID_MAX_ASPECT_RATIO = float(os.getenv("REID_MAX_ASPECT_RATIO", "1.3"))

# Optional blur/quality gate (variance of the Laplacian - a standard, cheap
# sharpness heuristic). Off by default for the PoC: video-compression softness
# on a legitimate, well-framed person crop can trip a naive blur threshold, and
# we would rather validate the appearance-similarity approach first before
# adding a second rejection axis. Turn on with REID_BLUR_CHECK=1 to experiment.
REID_BLUR_CHECK_ENABLED = _bool("REID_BLUR_CHECK", False)
REID_BLUR_MIN_VARIANCE = float(os.getenv("REID_BLUR_MIN_VARIANCE", "30.0"))


# ============================================================
# When to run Re-ID at all (never every frame)
# ============================================================

# A brand-new track collects this many valid observations, spaced apart by
# REID_BOOTSTRAP_INTERVAL_FRAMES confirmed frames of that SAME track, before
# its first identity match attempt. Spacing the observations out (rather than
# taking 5 back-to-back frames) buys a little pose/lighting diversity for the
# very first representative embedding.
REID_BOOTSTRAP_OBSERVATIONS = int(os.getenv("REID_BOOTSTRAP_OBSERVATIONS", "5"))
REID_BOOTSTRAP_INTERVAL_FRAMES = int(os.getenv("REID_BOOTSTRAP_INTERVAL_FRAMES", "3"))

# After a track has been matched/registered once, refresh its representative
# embedding at most this often - time-based, not frame-count-based, so it
# behaves the same regardless of the calling loop's inference rate (the same
# reasoning as zones.py's CROWD_DROP_GRACE_SECONDS: a frame-count interval
# means a different real-world cadence at every different FPS).
REID_UPDATE_INTERVAL_SECONDS = float(os.getenv("REID_UPDATE_INTERVAL_SECONDS", "30.0"))


# ============================================================
# Matching
# ============================================================

# PROVISIONAL - NOT VALIDATED. Raised from an original 0.6 (then 0.75, now
# 0.80) as an increasingly conservative PoC starting point after same-camera
# false merges were observed at 0.63-0.70 (see REID_CAMERA_MODE below - that
# policy is the REAL fix for those; this threshold is a second, independent
# lever, not a substitute for it). Still not a number pulled from any
# validated distribution - a placeholder until Phase 4's analyze mode
# (analyze_similarity.py) has produced real same-person vs different-person
# cosine-similarity distributions FROM THIS DEPLOYMENT's cameras, lighting and
# OSNet-AIN model. Re-run the analysis and update this before trusting any
# MATCH/NEW decision in a real report. Do not report 0.80 as production-ready.
REID_SIMILARITY_THRESHOLD = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.75"))

# How many of a global identity's most recent representative embeddings are
# kept and matched against ("a small gallery", not one embedding forever) -
# lets a global identity's stored appearance adapt to pose/lighting/camera
# changes without unbounded growth per person.
REID_GALLERY_SIZE = int(os.getenv("REID_GALLERY_SIZE", "5"))

# A candidate must beat the SECOND-best candidate by at least this much, on
# top of clearing REID_SIMILARITY_THRESHOLD, before it is accepted as a MATCH.
# This is what stops "P-0001 scored 0.68, P-0002 scored 0.64" from becoming a
# coin-flip MATCH to P-0001 - both cleared a lower bar in the past, but
# neither one is clearly THE match, so the honest answer is UNCERTAIN, not a
# forced guess. Only applies when 2+ candidates exist; a single candidate has
# nothing to be ambiguous against, so only the threshold applies.
REID_MIN_MATCH_MARGIN = float(os.getenv("REID_MIN_MATCH_MARGIN", "0.08"))


# ============================================================
# Cross-camera policy
#
# THE core fix for the false-merge problem: two independent local JeztSort
# tracks on the SAME camera must never be allowed to merge into one global
# identity through appearance matching alone. Two different people in similar
# clothing on one camera can legitimately score 0.63-0.70 - close enough to
# look tempting, nowhere near proof. OSNet's actual value is telling the SAME
# person apart across DIFFERENT cameras, where there is no local track
# continuity to lean on at all - so that is what it is used for here.
# ============================================================

# SUPERSEDED by REID_CAMERA_MODE below (kept only so old scripts/env files
# that still set it are not silently ignored without an obvious constant to
# grep for) - GlobalReIDManager no longer reads this to make decisions.
# REID_CAMERA_MODE="cross_camera_only" (the default) reproduces exactly the
# behaviour this flag used to control when it was True.
REID_CROSS_CAMERA_ONLY = _bool("REID_CROSS_CAMERA_ONLY", True)

# A stored observation older than this many seconds is excluded from
# candidate matching (though never deleted - old history is still readable
# via Store.get_history() for reporting). This stops a person seen once,
# hours ago, on some far camera from being treated as an immediate camera
# transition just because the appearance happens to be similar. 300s (5 min)
# is a PoC starting point, not a measured value - a real facility's realistic
# camera-to-camera walking time should set this once known.
REID_MAX_TIME_GAP_SECONDS = float(os.getenv("REID_MAX_TIME_GAP_SECONDS", "300"))

# Optional camera-topology restriction, OFF by default. When enabled, a
# camera's cross-camera candidates are further narrowed to only the cameras
# listed as reachable from it below - e.g. do not even consider a match
# against a camera nobody could plausibly have walked to yet. CAMERA_TRANSITIONS
# is intentionally EMPTY: the real physical adjacency of the 24-camera estate
# is not known to this PoC, and inventing one would silently bias matching on
# a fabricated assumption. Populate it (camera_id -> set of reachable
# camera_ids) and flip REID_USE_CAMERA_TOPOLOGY=1 once the real topology is
# known; a camera_id missing from the map is treated as "no restriction" (not
# "match nothing"), so a partially-filled map never blocks the cameras you
# have not gotten to yet.
REID_USE_CAMERA_TOPOLOGY = _bool("REID_USE_CAMERA_TOPOLOGY", False)
CAMERA_TRANSITIONS = {
    # "CAM-R01": {"CAM-R02", "CAM-R09"},
}


# ============================================================
# EXPERIMENTAL: same-camera Re-ID (JeztSort recovery layer)
#
# The cross-camera policy above exists because two DIFFERENT people on one
# camera can score 0.63-0.70 - not proof of anything. But appearance
# similarity may still be useful WITHIN one camera for a narrower purpose:
# recovering JeztSort's own local-track churn (a person briefly lost to
# occlusion/confidence dropout reappears as a brand-new local track id, e.g.
# P-001 -> disappears -> P-019 is really the same physical person). This does
# NOT replace JeztSort and NEVER writes back into TrackManager - it only
# gives the Re-ID layer a second, appearance-based opinion about whether two
# local track ids on the same camera are the same person.
#
# NOT enabled by default. Enabling this reintroduces the same-camera false-
# merge risk unless the margin/conflict/time-gap rules below are respected -
# it is an experiment to run and observe with REID_MODE=analyze, not a
# validated production capability.
# ============================================================

# "cross_camera_only"  (DEFAULT - unchanged existing behaviour) same-camera
#                       candidates are never even considered; see above.
# "same_camera"         EXPERIMENTAL - only same-camera recovery candidates
#                       are considered; cross-camera matching is off.
# "same_and_cross"      EXPERIMENTAL - both pools are searched and the
#                       single best candidate across either one wins.
# Validated at import time - a typo here fails loudly rather than silently
# falling back to some other mode.
REID_CAMERA_MODE = os.getenv("REID_CAMERA_MODE", "cross_camera_only").strip().lower()

_VALID_CAMERA_MODES = ("cross_camera_only", "same_camera", "same_and_cross")
if REID_CAMERA_MODE not in _VALID_CAMERA_MODES:
    raise ValueError(
        f"REID_CAMERA_MODE={REID_CAMERA_MODE!r} is not valid - "
        f"must be one of {_VALID_CAMERA_MODES}"
    )

# The critical safety rail for "same_camera"/"same_and_cross": two different
# people simultaneously visible on one camera (local track A -> P-0001, local
# track B -> P-0002) must not have a THIRD local track (JeztSort having lost
# and recreated A, say) matched onto P-0001 or P-0002 just because one of them
# has the highest historical similarity - both are ACTIVELY represented by
# another local track right now, which a genuine recovery candidate never is.
# When True, such a same-camera candidate is excluded from matching (the
# track still gets its own new global_person_id; see SAME_CAM_BLOCKED in
# reid_manager.py). Configurable per the "make this configurable if possible"
# requirement - False removes this rail entirely, for deliberate A/B testing
# of how much it actually matters, not a recommended running mode.
REID_SAME_CAMERA_ACTIVE_CONFLICT = _bool("REID_SAME_CAMERA_ACTIVE_CONFLICT", True)

# How recently a local track must have been CONFIRMED (touched by
# needs_observation(), so this updates every tick the track is seen,
# independent of whether Re-ID/OSNet ran) for its resolved identity to count
# as "actively represented" on that camera for the conflict check above.
# NEW default, documented here as required: 8 seconds. Reasoning - long
# enough to absorb a several-frame confirmation gap (occlusion, a confidence
# dip) without a person's identity losing "active" status the instant JeztSort
# has one bad frame, short enough that a track truly gone from the scene stops
# blocking a legitimate recovery match within a few seconds, not indefinitely.
# Not measured from real footage - a PoC placeholder like every other
# threshold on this page, tune it once same-camera analyze-mode data exists.
REID_ACTIVE_WINDOW_SECONDS = float(os.getenv("REID_ACTIVE_WINDOW_SECONDS", "8.0"))


# ============================================================
# EXPERIMENTAL: camera groups / location-specific Re-ID policy
#
# Not a camera-transition/topology-timing model - deliberately NOT
# implemented yet (no travel-time windows, no adjacency-with-timing, no
# trajectory prediction; that is CAMERA_TRANSITIONS/REID_USE_CAMERA_TOPOLOGY
# above, plus future work). This is a simpler, flatter idea: which cameras
# are even candidates for one another, and what threshold/margin applies,
# can differ per physical location - the two cameras in one small room may
# reasonably need a different (often lower) bar than two cameras across a
# large, busier facility, because appearance conditions (lighting, crowd
# density, camera angle) genuinely differ per location. A camera belongs to
# AT MOST one group; an unconfigured camera falls back to the global
# REID_SIMILARITY_THRESHOLD/REID_MIN_MATCH_MARGIN unchanged - existing
# behaviour for every camera not deliberately opted into a group.
#
# Dashboard-ready by construction: if REID_CAMERA_GROUPS_FILE exists, its
# JSON content REPLACES the inline REID_CAMERA_GROUPS dict below entirely (a
# future dashboard writes that file; nothing here needs to change to pick it
# up). Group name / camera membership / threshold / margin are pure data,
# never referenced by name anywhere in reid_manager.py's decision logic.
# ============================================================

REID_CAMERA_GROUPS_FILE = os.getenv(
    "REID_CAMERA_GROUPS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "camera_groups.json"),
)

# Inline default - used only when REID_CAMERA_GROUPS_FILE does not exist.
# One real group populated for the two cameras this PoC is actively being
# tested on (see README "Camera Group / Location-specific Re-ID Policy") -
# NOT a claim that 0.75/0.08 are validated for this location; they currently
# match the global defaults on purpose, so enabling this group changes
# nothing about today's live behaviour and only proves the resolution
# mechanism itself works end-to-end. A real dashboard-configured facility
# would tune these independently per location once real data exists.
_REID_CAMERA_GROUPS_INLINE = {
    "location_a": {
        "cameras": ["CAM-R09", "CAM-R10"],
        "similarity_threshold": 0.75,
        "min_match_margin": 0.08,
    },
}

# If set to a configured group name, a camera with NO group of its own
# borrows that group's threshold/margin instead of the bare global defaults
# (still does NOT add the camera to that group's allowed_cameras list - an
# unconfigured camera never gains cross-camera candidate access to a group it
# was not explicitly placed in; only its threshold/margin fall back to it).
# None (default) - an unconfigured camera uses REID_SIMILARITY_THRESHOLD /
# REID_MIN_MATCH_MARGIN directly, exactly as before this feature existed.
REID_DEFAULT_GROUP = os.getenv("REID_DEFAULT_GROUP") or None


def _normalize_camera_id(camera_id):
    return camera_id.strip().upper()


def _validate_and_normalize_camera_groups(groups, default_group):
    """
    Fails LOUDLY at import time (raises), never silently - a broken camera-
    group config must not quietly fall back to "no groups" or "no filtering".
    """
    if not isinstance(groups, dict):
        raise ValueError(f"REID_CAMERA_GROUPS must be a dict of group_name -> config, got {type(groups)}")

    seen_cameras = {}
    normalized = {}

    for group_name, group_cfg in groups.items():
        if not isinstance(group_cfg, dict):
            raise ValueError(f"REID_CAMERA_GROUPS[{group_name!r}] must be a dict")

        cameras = group_cfg.get("cameras")
        if not cameras:
            raise ValueError(f"REID_CAMERA_GROUPS[{group_name!r}]: 'cameras' must be a non-empty list")

        normalized_cameras = [_normalize_camera_id(c) for c in cameras]

        for cam in normalized_cameras:
            if cam in seen_cameras:
                raise ValueError(
                    f"REID_CAMERA_GROUPS: camera {cam!r} appears in both "
                    f"{seen_cameras[cam]!r} and {group_name!r} - a camera can only belong to one group"
                )
            seen_cameras[cam] = group_name

        threshold = group_cfg.get("similarity_threshold")
        if threshold is not None and not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"REID_CAMERA_GROUPS[{group_name!r}]: similarity_threshold must be in [0, 1], got {threshold}"
            )

        margin = group_cfg.get("min_match_margin")
        if margin is not None and margin < 0:
            raise ValueError(
                f"REID_CAMERA_GROUPS[{group_name!r}]: min_match_margin must be >= 0, got {margin}"
            )

        # Optional per-group override of REID_CAMERA_MODE. None/absent ->
        # caller falls back to the manager-wide camera_mode, exactly like
        # threshold/margin above - a group only needs to set this when its
        # policy genuinely differs from the rest of the deployment.
        camera_mode = group_cfg.get("camera_mode") or None
        if camera_mode is not None and camera_mode not in _VALID_CAMERA_MODES:
            raise ValueError(
                f"REID_CAMERA_GROUPS[{group_name!r}]: camera_mode={camera_mode!r} is not valid - "
                f"must be one of {_VALID_CAMERA_MODES} (or omitted to use the global default)"
            )

        normalized[group_name] = {
            "cameras": normalized_cameras,
            "similarity_threshold": threshold,   # None -> caller falls back to the global default
            "min_match_margin": margin,          # None -> caller falls back to the global default
            "camera_mode": camera_mode,          # None -> caller falls back to the global default
        }

    if default_group is not None and default_group not in normalized:
        raise ValueError(
            f"REID_DEFAULT_GROUP={default_group!r} does not match any configured "
            f"REID_CAMERA_GROUPS group name ({sorted(normalized)})"
        )

    return normalized


def _load_camera_groups():
    if os.path.exists(REID_CAMERA_GROUPS_FILE):
        with open(REID_CAMERA_GROUPS_FILE) as handle:
            data = json.load(handle)
        return data.get("groups", data)   # accept either {"groups": {...}} or a bare {...}

    return _REID_CAMERA_GROUPS_INLINE


REID_CAMERA_GROUPS = _validate_and_normalize_camera_groups(_load_camera_groups(), REID_DEFAULT_GROUP)


# ============================================================
# Re-ID camera AUTHORIZATION (production-hardening P0 fix)
#
# REID_CAMERA_GROUPS answers "which cameras may this camera match against."
# It has never answered "is this camera authorized to run Re-ID at all" -
# a camera absent from every group has always fallen through to
# GlobalReIDManager's bare global defaults with NO restriction (allowed_
# cameras=None), including cameras the dashboard operator never touched.
# REID_ENABLED_CAMERAS is the fix: the authorization set. A camera_id NOT
# in this set is NOT authorized - GlobalReIDManager.needs_observation()
# refuses to even bootstrap it (see reid_manager.py), never mind search or
# store anything.
#
# Fully backward compatible with the reid_poc PoC/test-harness (non-
# dashboard) path: with no dashboard involved, this defaults to exactly the
# union of every configured group's own camera list - every camera anyone
# already bothered to put in a group is authorized, and nothing else is,
# which is the safe, conservative behaviour requested (never silently widen
# to "everything"). A production dashboard fetch (see reid_config_provider.py)
# later REPLACES this with the dashboard's own authoritative, possibly-wider
# set (grouped AND ungrouped-but-individually-enabled cameras) - see
# _validate_and_normalize_enabled_cameras() below for the one invariant that
# is still enforced even then: every camera inside a configured group must
# also be authorized (a dashboard payload that violates this is malformed -
# rejected wholesale, not silently patched around).
# ============================================================

def _normalize_enabled_cameras(enabled_cameras):
    if not isinstance(enabled_cameras, (list, tuple, set, frozenset)):
        raise ValueError(
            f"enabled_cameras must be a list of camera_id strings, got {type(enabled_cameras)}"
        )
    return frozenset(_normalize_camera_id(c) for c in enabled_cameras)


def _validate_and_normalize_enabled_cameras(enabled_cameras, groups):
    """
    groups is an ALREADY-normalized REID_CAMERA_GROUPS-shaped dict (i.e. the
    return value of _validate_and_normalize_camera_groups()) - every camera
    appearing in any group's "cameras" list MUST also appear in
    enabled_cameras. A dashboard payload that lists a camera as a group
    member but not as reid_enabled would be internally inconsistent (see
    dashboard cameras/api_views.py's _reid_groups_payload(), which only ever
    admits reid_enabled=True cameras into a group's list in the first place)
    - failing loudly here catches that bug at the config boundary, rather
    than letting it silently degrade into "that one camera just never gets
    Re-ID" with no explanation.
    """
    normalized = _normalize_enabled_cameras(enabled_cameras)

    grouped_cameras = {cam for group_cfg in groups.values() for cam in group_cfg["cameras"]}
    inconsistent = grouped_cameras - normalized
    if inconsistent:
        raise ValueError(
            f"REID_ENABLED_CAMERAS is missing {sorted(inconsistent)}, which are configured as "
            f"members of a REID_CAMERA_GROUPS group - every grouped camera must also be enabled"
        )

    return normalized


def _default_enabled_cameras(groups):
    """The safe default when no dashboard has ever supplied an explicit
    enabled-cameras set: exactly the cameras already placed in a group.
    Never wider than that - an ungrouped camera stays unauthorized until a
    dashboard fetch explicitly says otherwise."""
    return frozenset(cam for group_cfg in groups.values() for cam in group_cfg["cameras"])


def _load_enabled_cameras():
    """Optional explicit "enabled_cameras" key in the same JSON file
    REID_CAMERA_GROUPS_FILE already loads (for forward-compatibility/
    explicitness in a hand-edited static file) - absent by default, in
    which case the safe default above is used instead."""
    if os.path.exists(REID_CAMERA_GROUPS_FILE):
        with open(REID_CAMERA_GROUPS_FILE) as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "enabled_cameras" in data:
            return data["enabled_cameras"]
    return None


_explicit_enabled_cameras = _load_enabled_cameras()
REID_ENABLED_CAMERAS = (
    _validate_and_normalize_enabled_cameras(_explicit_enabled_cameras, REID_CAMERA_GROUPS)
    if _explicit_enabled_cameras is not None
    else _default_enabled_cameras(REID_CAMERA_GROUPS)
)


# ============================================================
# SAME-CAMERA RECOVERY ROLLOUT SET
#
# WHAT PROBLEM THIS SOLVES
#   Measured on one production run's log: 1,730 SKIPPED_SAME_CAMERA
#   decisions against 71 MATCH. SKIPPED_SAME_CAMERA means a same-camera
#   identity CLEARED the camera's similarity threshold and was discarded
#   anyway, because production resolves cross_camera_only (see
#   REID_CAMERA_MODE above). Per camera: CAM-R09 704, CAM-R10 629,
#   CAM-R16 397 - and CAM-R16 recorded ZERO accepted matches in the whole
#   run, so every one of its 397 was a recovery thrown away.
#
# WHAT THIS SET DOES
#   A camera named here resolves its camera_mode from its GROUP MEMBERSHIP
#   instead of the deployment-wide default (see reid_manager._policy_for()):
#       grouped   -> "same_and_cross"   same-camera recovery, existing
#                                       cross-camera group behaviour intact
#       ungrouped -> "same_camera"      camera-local only; cross-camera
#                                       matching is structurally impossible
#   A camera NOT named here is completely unaffected and keeps whatever
#   camera_mode it resolves today.
#
# WHY AN EXPLICIT SET RATHER THAN A DEPLOYMENT-WIDE SWITCH
#   The rule above is the intended end state for every Re-ID camera, but it
#   changes which identities get created, so it is rolled out per camera
#   against measured evidence rather than turned on everywhere at once.
#   CAM-R16 is the validation target precisely because its 397/0 split makes
#   any change unambiguous. Empty by default: this file ships inert, and a
#   deployment opts cameras in explicitly.
#
# PRECEDENCE. A group that sets its OWN camera_mode in the dashboard still
# wins over this set - an explicit operator choice is never overridden by a
# rollout list (see reid_manager._policy_for()).
# ============================================================

REID_SAME_CAMERA_CAMERAS = frozenset(
    _normalize_camera_id(c)
    for c in os.getenv("REID_SAME_CAMERA_CAMERAS", "").split(",")
    if c.strip()
)


# ============================================================
# Local track-state lifecycle (production-hardening P1 fix)
#
# GlobalReIDManager._tracks retains one _TrackState per DISTINCT
# (camera_id, local_track_id) EVER SEEN, for the life of the process - the
# only eviction path is forget_camera(), an explicit whole-camera reset, not
# a per-track prune. On a continuously-running CCTV system this grows
# without bound. This TTL bounds it: a track not confirmed (needs_
# observation() not called for it) in longer than this many seconds is
# evicted - the GLOBAL identity/gallery data in the Store is completely
# untouched by this (different lifetime, different owner - see
# GlobalReIDManager.evict_stale_tracks()'s own docstring).
#
# Default is DERIVED, not invented: REID_MAX_TIME_GAP_SECONDS already
# defines "how old can a stored observation be and still count as a fresh
# candidate" - once a local track has been unconfirmed for longer than that,
# its own bootstrap state is not contributing anything a fresh re-bootstrap
# (via ordinary appearance matching against the still-intact gallery)
# would not already achieve on its own. Using the SAME number keeps one
# mental model ("how long is an absence still 'recent'") instead of two
# independently-tuned ones.
REID_TRACK_STATE_TTL_SECONDS = float(
    os.getenv("REID_TRACK_STATE_TTL_SECONDS", str(REID_MAX_TIME_GAP_SECONDS))
)

# How often the cleanup sweep runs - NOT a per-frame scan. Piggybacks on the
# same "track elapsed time, only act once due" pattern reid_adapter.py's own
# _maybe_print_stats() already uses for REID_STATS_INTERVAL_SECONDS. This
# number (unlike the TTL above) is a plain performance/responsiveness
# parameter, not derived from an existing constant: frequent enough that
# _tracks never grows more than about one interval's worth of traffic past
# its steady-state bound, infrequent enough that scanning the whole dict
# costs nothing measurable at this cadence (a single pass over a dict of
# even tens of thousands of entries is sub-millisecond).
REID_TRACK_CLEANUP_INTERVAL_SECONDS = float(os.getenv("REID_TRACK_CLEANUP_INTERVAL_SECONDS", "60.0"))


# ============================================================
# Asynchronous Store persistence (production-hardening P1 fix)
#
# reid_manager.AsyncUpsertStore wraps any Store so upsert() never blocks the
# calling (inference) thread on network I/O - see that class's own
# docstring for the full design. search() is UNCHANGED and stays
# synchronous: the calling decision genuinely needs its result (this is a
# correctness requirement, not a missed optimisation - see the module
# docstring in reid_manager.py).
# ============================================================

#: Bounded - never grows without limit even if persistence falls behind
#: search/decision throughput. At the observed production decision rate
#: (a few per second per camera, never per-frame), this comfortably absorbs
#: several seconds of a slow/stalled Qdrant without dropping anything in
#: normal operation, while still bounding worst-case memory if Qdrant is
#: down for a while.
REID_ASYNC_UPSERT_QUEUE_SIZE = int(os.getenv("REID_ASYNC_UPSERT_QUEUE_SIZE", "500"))

#: A failed upsert is retried this many times (linear backoff) before being
#: dropped and logged - persistence failures must never crash inference or
#: block the writer thread indefinitely on one bad write.
REID_ASYNC_UPSERT_MAX_RETRIES = int(os.getenv("REID_ASYNC_UPSERT_MAX_RETRIES", "3"))
REID_ASYNC_UPSERT_RETRY_BACKOFF_SECONDS = float(os.getenv("REID_ASYNC_UPSERT_RETRY_BACKOFF_SECONDS", "0.5"))

#: On shutdown, how long to wait for the queue to drain before giving up and
#: closing anyway - shutdown must be bounded, never indefinite.
REID_ASYNC_UPSERT_SHUTDOWN_FLUSH_SECONDS = float(os.getenv("REID_ASYNC_UPSERT_SHUTDOWN_FLUSH_SECONDS", "5.0"))


# ============================================================
# Synchronous search retry (production-hardening G-1 fix)
#
# search() stays synchronous and unretried-by-default at the Store layer -
# GlobalReIDManager._search_with_retry() wraps the specific search() calls a
# bootstrap DECISION makes (never a REFRESH, which already self-heals - see
# that method's own docstring) with a small bounded retry, so a single
# transient Qdrant blip does not permanently strand the local track that
# happened to finish bootstrapping at that exact moment (CONFIRMED finding
# G-1 - reproduced against a real Qdrant outage, see the validation report).
# ============================================================

#: Total attempts (the first try PLUS retries), not "retries on top of the
#: first try" - deliberately named/worded to match this fix's own spec
#: ("max 2 or 3 attempts total"), not REID_ASYNC_UPSERT_MAX_RETRIES's
#: different counting convention. 3 total attempts, short linear backoff
#: between them, bounds worst-case added latency to a few hundred
#: milliseconds - never a long block of the inference pipeline.
REID_SEARCH_RETRY_MAX_ATTEMPTS = int(os.getenv("REID_SEARCH_RETRY_MAX_ATTEMPTS", "3"))

#: Linear backoff unit between search attempts (same style as
#: REID_ASYNC_UPSERT_RETRY_BACKOFF_SECONDS, scaled down - a search retry
#: sits ON the inference thread, unlike an async upsert retry, so this must
#: stay small). Default 50ms: attempt 2 waits 50ms, attempt 3 waits 100ms -
#: 150ms worst-case total added latency before giving up.
REID_SEARCH_RETRY_BACKOFF_SECONDS = float(os.getenv("REID_SEARCH_RETRY_BACKOFF_SECONDS", "0.05"))


# ============================================================
# Store backend - swap with ZERO changes to reid_manager.py
# ============================================================

# "memory"  - InMemoryReIDStore (Phase 2): brute-force cosine in a Python
#             dict, gone when the process exits. Validate matching logic here
#             first.
# "qdrant"  - QdrantReIDStore (Phase 3): persistent, the collection below.
REID_STORE = os.getenv("REID_STORE", "memory").strip().lower()

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cctv_person_reid")
QDRANT_VECTOR_DIM = EMBEDDING_DIM
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "COSINE")

# "server" talks to a running Qdrant instance (host/port below); "local" uses
# qdrant-client's embedded on-disk mode at QDRANT_LOCAL_PATH - no server
# process needed at all, useful for a quick offline check.
QDRANT_MODE = os.getenv("QDRANT_MODE", "server").strip().lower()
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6343"))
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "reid_poc/qdrant_local_data")


# ============================================================
# Test harness (test_reid_tracking.py)
# ============================================================

TARGET_INFERENCE_FPS = float(os.getenv("TARGET_INFERENCE_FPS", "10.0"))

# "live"    - normal operation: resolve global ids, print the status table.
# "analyze" - Phase 4: log every computed similarity (with a structural
#             same/different-track label) instead of gating on the threshold,
#             for analyze_similarity.py to summarise afterwards.
REID_MODE = os.getenv("REID_MODE", "live").strip().lower()
ANALYZE_LOG_PATH = os.getenv("ANALYZE_LOG_PATH", "reid_poc/similarity_log.jsonl")
ANALYZE_SAVE_CROPS = _bool("ANALYZE_SAVE_CROPS", True)
ANALYZE_CROPS_DIR = os.getenv("ANALYZE_CROPS_DIR", "reid_poc/analyze_crops")

STATS_INTERVAL_SECONDS = float(os.getenv("STATS_INTERVAL_SECONDS", "5.0"))

# Live imshow mosaic - one tile per camera, boxes coloured by Re-ID status,
# labelled with the local JeztSort id, the resolved global_person_id,
# similarity and status (or bootstrap progress while still collecting
# observations), plus a shared banner with cross-camera totals. Off by
# default is NOT the default here (this is the PoC's main visual aid) but
# turn it off for a headless run - unattended REID_MODE=analyze collection,
# or scaling CAMERAS up towards all 24 - where a display window is either
# unavailable or just overhead you don't want.
SHOW_WINDOW = _bool("SHOW_WINDOW", True)
WINDOW_TILE_WIDTH = int(os.getenv("WINDOW_TILE_WIDTH", "1080"))
WINDOW_TILE_HEIGHT = int(os.getenv("WINDOW_TILE_HEIGHT", "960"))
WINDOW_COLS = int(os.getenv("WINDOW_COLS", "0"))   # 0 = auto (roughly square)
