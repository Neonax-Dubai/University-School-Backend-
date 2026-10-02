"""
Abandoned (unattended) object detection - one alert per PHYSICAL unattended episode.  [RFP 6.1]

An object of an enabled class (backpack / handbag / suitcase) is UNATTENDED when it has stayed
STATIONARY with NO PERSON NEARBY continuously for longer than a dwell threshold. One physical
object left in one place is one EPISODE, and one episode raises exactly ONE alert, however many
tracker ids the object passes through while it lies there.

Why it needs tracking: "stationary" and "for N seconds" are only decidable across frames, so this
sits on top of the JeztSort object-group tracks (O-####) exactly the way zones.py sits on top of
person / vehicle tracks. But a TRACK ID IS NOT AN OBJECT IDENTITY - see the episode section below.

The two-condition rule is what separates a real left-behind bag from a bag its owner is standing
beside: while a person is within the proximity radius the dwell clock is held at zero, so only an
object whose owner has walked away and STAYED away trips the alarm. It also catches an object that
was already sitting there when the camera came up, because the clock starts the first frame it is
seen stationary with nobody near it.

A person detector can miss a frame or two even while the owner is standing right there - a turn, a
brief occlusion. ABANDON_OWNER_GRACE_SECONDS absorbs a short window of apparent non-attendance
before the object counts as genuinely unattended at all; only once that elapses does the dwell
clock start. Grace time is real seconds, not a frame count.

STATE IS KEPT PER RESTING SPOT, NOT PER TRACK ID  (Dubai, 2026-09-15)
--------------------------------------------------------------------
This used to key its clock on the track id, which made a real test on CAM-R25 impossible to trip.
A black backpack left against a wall was detected at 0.26-0.58 confidence - either side of the
camera's 0.45 floor - so it dropped out of the detections for 2.5-10 s at a time. tracking.py gives
a track that has coasted for more than 1.5 s a NEW label when it is matched again, so every dropout
handed this module a brand-new track id. A left bag does not move, so its POSITION is the identity
that survives a detector dropout, and attendance is judged at that position on every frame,
including frames where the bag itself is missed (people are detected independently of the bag).

ZAYED: ONE ALERT PER PHYSICAL EPISODE  (2026-10-02)
---------------------------------------------------
The spot fixed the CLOCK, not the ALERT. On 2026-09-30 C101 raised 24 "Unattended Object" alarms
in 1 h 40 min for a handful of bags (inference log, 10:26-12:05Z):
  * camera_02 backpack O-101W-0877 alarmed SEVEN times without ever changing track id, and
    camera_01 backpack O-101W-1064 three times: every person passing within the proximity radius
    reset the spot - `alerted` included - so the next 32 s with nobody near alarmed again;
  * camera_01 handbag O-101W-0940 -> 0956 -> 0975 -> 0994 -> 1011: one alarm per id. The bag,
    lying in plain view, was detected at 0.25-0.57 and missed for up to 70 s at a time with nobody
    near it; a spot unseen for more than 30 s was forgotten, so each return started over;
  * events.py debounced per TRACK id (a new id bypassed it) and the dashboard raised one alarm per
    event, so nothing downstream folded the repeats.
Each resting object is now an EPISODE with a lifecycle:

    OBSERVED    tracked, attended (or just appeared)
    CANDIDATE   nobody near: the owner-grace window is running
    UNATTENDED  the dwell clock is running
    ALERTED     the dwell completed: the ONE alert of this episode has been raised
    RESOLVED    removed / moved / attended - final. An alerted episode reports its resolution once.

Association (a detection joins an episode), in this order - always camera-local:
  1. track continuity: the episode its track id already belongs to;
  2. else an unresolved episode of this camera that is not seen this frame, within
     ASSOCIATION_DISTANCE x the box diagonal of its resting position, of a similar box size
     (ASSOCIATION_SIZE_RATIO), of a compatible class, and not yet past the recovery window;
  3. else a NEW episode.
A second detection lying on an episode already seen this frame is a fragment of it, not a new one.

Resolution is deterministic and never caused by one frame:
  * removed  - the object has not been seen for RECOVERY_WINDOW seconds while its spot was in view
               (time with a person at the spot does not count: they may simply be in front of it),
               or for MAX_UNSEEN_SECONDS at all;
  * moved    - the object's position left the move tolerance for MOVE_CONFIRM_SECONDS while a
               person was at its spot (MOVE_HANDLING_SECONDS): a resting object only moves when
               someone handles it, so a box that wanders with nobody there is an artefact, not a
               move. It is somewhere else now: the resting episode is over and a new one starts there;
  * attended - an ALERTED object has had a person beside it continuously (gaps under the grace
               window) for ATTENDED_RESOLVE_SECONDS: someone has taken charge of it.
Person proximity has hysteresis: a person briefly passing an ALERTED object neither closes it nor,
when they leave, raises it again. Before the alert the original rule stands: a person near the
object holds its clock (after PREALERT_ATTEND_SECONDS of attendance, default 0 = immediately).
There is no global cooldown: a different object, or this one after its episode has resolved, alerts.

Env overrides (all validated against the 2026-09-30 C101 recordings - see
docs/ZAYED_UNATTENDED_OBJECT.md):
  ABANDON_DWELL_SECONDS            unattended-stationary time before alert          (default 30)
  ABANDON_MOVE_TOLERANCE           max centre drift still counted as 'still', fraction of the
                                   object's diagonal                                 (default 0.5)
  ABANDON_MOVE_CONFIRM_SECONDS     a drift must persist this long to count as a move (default 1.0)
  ABANDON_MOVE_HANDLING_SECONDS    ... and a person must have been at the spot within this long
                                   before it                                         (default 10)
  ABANDON_PROXIMITY                person-near radius, fraction of frame diagonal    (default 0.12)
  ABANDON_REQUIRE_OWNERLESS        1 = require nobody near, 0 = pure stationary      (default 1)
  ABANDON_OWNER_GRACE_SECONDS      grace before "no owner" starts the dwell clock; also the gap
                                   that still counts as one continuous attendance     (default 2.0)
  ABANDON_PREALERT_ATTEND_SECONDS  attendance that holds a NOT yet alerted clock       (default 0)
  ABANDON_ATTENDED_RESOLVE_SECONDS continuous attendance that resolves an ALERTED one (default 300)
  ABANDON_RECOVERY_WINDOW_SECONDS  unseen-in-view time before an episode is 'removed'; the window
                                   in which a returning detection continues it
                                   (default 120; ABANDON_SPOT_MEMORY_SECONDS is the old name)
  ABANDON_MAX_UNSEEN_SECONDS       unseen at all, even behind people, before 'removed' (default 600)
  ABANDON_ASSOCIATION_DISTANCE     max centre distance for a new track to continue an episode,
                                   fraction of the larger box diagonal              (default 0.75)
  ABANDON_ASSOCIATION_SIZE_RATIO   max box-area ratio for that                        (default 4.0)
  ABANDON_ASSOCIATION_STRICT_CLASS 1 = only the same class continues an episode       (default 0:
                                   the detector reports one bag as backpack AND handbag)
  ABANDON_CLASSES                  unattended classes, names or COCO ids, e.g. "backpack,handbag";
                                   only SUPPORTED_CLASSES are accepted (ABANDON_CLASS_IDS is the old
                                   name)                                  (default backpack,handbag,suitcase)
  ABANDON_MIN_OBSERVED_FRACTION    share of the dwell the object must actually be DETECTED (0.3)
  ABANDON_OBJECT_CONFIDENCE        detector confidence a candidate bag needs (default 0.25)
"""
import itertools
import math
import os
import random
import time


def _env_float(name, default, *fallback_names):
    for key in (name,) + fallback_names:
        value = os.getenv(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                print(f"[UNATTENDED] ignoring {key}={value!r}: not a number", flush=True)
    return float(default)


DWELL_SECONDS = _env_float("ABANDON_DWELL_SECONDS", 30)
MOVE_TOLERANCE_FRAC = _env_float("ABANDON_MOVE_TOLERANCE", 0.5)
MOVE_CONFIRM_SECONDS = _env_float("ABANDON_MOVE_CONFIRM_SECONDS", 1.0)
MOVE_HANDLING_SECONDS = _env_float("ABANDON_MOVE_HANDLING_SECONDS", 10.0)
PROXIMITY_FRAC = _env_float("ABANDON_PROXIMITY", 0.12)
REQUIRE_OWNERLESS = os.getenv("ABANDON_REQUIRE_OWNERLESS", "1") == "1"
GRACE_SECONDS = _env_float("ABANDON_OWNER_GRACE_SECONDS", 2.0)
PREALERT_ATTEND_SECONDS = _env_float("ABANDON_PREALERT_ATTEND_SECONDS", 0.0)
ATTENDED_RESOLVE_SECONDS = _env_float("ABANDON_ATTENDED_RESOLVE_SECONDS", 300.0)
RECOVERY_WINDOW_SECONDS = _env_float("ABANDON_RECOVERY_WINDOW_SECONDS", 120.0, "ABANDON_SPOT_MEMORY_SECONDS")
MAX_UNSEEN_SECONDS = _env_float("ABANDON_MAX_UNSEEN_SECONDS", 600.0)
ASSOCIATION_DISTANCE_FRAC = _env_float("ABANDON_ASSOCIATION_DISTANCE", 0.75)
ASSOCIATION_SIZE_RATIO = _env_float("ABANDON_ASSOCIATION_SIZE_RATIO", 4.0)
ASSOCIATION_STRICT_CLASS = os.getenv("ABANDON_ASSOCIATION_STRICT_CLASS", "0") == "1"

#: Kept for callers of the Dubai name: how long a resting spot survives its object not being seen.
SPOT_MEMORY_SECONDS = RECOVERY_WINDOW_SECONDS

# ---- classes -----------------------------------------------------------------------------------
#: The classes this use case supports: carried objects the COCO detector reports and the use case
#: has been validated for. "luggage" is COCO "suitcase". COCO has NO box / package class, so the
#: deployed detector cannot report one and it cannot be enabled here - that needs a custom model.
SUPPORTED_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}
CLASS_ALIASES = {"backpack": 24, "handbag": 26, "suitcase": 28, "luggage": 28}
DEFAULT_CLASSES = "backpack,handbag,suitcase"


def parse_classes(spec, log=print):
    """Class ids from 'backpack,handbag' / '24,26' / a mix. Unsupported entries are refused, never
    enabled; if nothing valid remains, the validated default applies (and it says so)."""
    ids, refused = set(), []
    for token in str(spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        class_id = int(token) if token.isdigit() else CLASS_ALIASES.get(token.lower())
        if class_id in SUPPORTED_CLASSES:
            ids.add(class_id)
        else:
            refused.append(token)
    if refused:
        log(f"[UNATTENDED] refusing unsupported unattended-object classes {refused}; "
            f"supported: {sorted(set(SUPPORTED_CLASSES.values()))}")
    if not ids:
        if str(spec or "").strip():
            log(f"[UNATTENDED] no supported class configured - using the default {DEFAULT_CLASSES}")
        ids = {CLASS_ALIASES[name] for name in DEFAULT_CLASSES.split(",")}
    return frozenset(ids)


CANDIDATE_CLASS_IDS = parse_classes(os.getenv("ABANDON_CLASSES") or os.getenv("ABANDON_CLASS_IDS")
                                    or DEFAULT_CLASSES)

# Confidence floor for candidate bags on a camera with abandoned_object armed. A still black bag
# against a wall scored 0.26-0.58 on CAM-R25, either side of the camera's 0.45 floor, and every dip
# was a dropout; candidate classes pass at this floor into the tracker and this analytic only.
OBJECT_CONFIDENCE = _env_float("ABANDON_OBJECT_CONFIDENCE", 0.25)

# Spot continuity lets the clock run across detector dropouts, so something must stop a spot that is
# barely ever seen from alarming: each observation credits the time since the previous one, capped at
# OBSERVATION_CREDIT_CAP_SECONDS, and the alarm needs this share of the dwell.
MIN_OBSERVED_FRACTION = _env_float("ABANDON_MIN_OBSERVED_FRACTION", 0.3)
OBSERVATION_CREDIT_CAP_SECONDS = 0.5

# Minimum centre drift (px) that still counts as 'still' regardless of box size, so detection jitter
# on a small far-away box does not count as a move.
MIN_MOVE_TOLERANCE_PX = 15.0

#: Two candidate boxes overlapping this much in one frame are one physical bag. At low confidence the
#: detector reports a bag as "backpack" AND "handbag" with near-identical boxes.
DUPLICATE_IOU = 0.5

#: A detection overlapping an episode already seen in the same frame at least this much (or centred
#: inside its box) is a fragment of that object - a partial box - not a second object.
FRAGMENT_IOU = 0.1

#: An interval between two updates of one camera longer than this is a stall, not observed absence:
#: only this much of it counts toward "unseen" (tracking.py treats stalls the same way).
MAX_FRAME_GAP_SECONDS = 2.0

#: A resolved episode is kept this long (reporting, and so its track ids stay bound to it).
RESOLVED_RETENTION_SECONDS = 300.0

# active_count() reports a clock as running only if its object was seen this recently.
ACTIVE_FRESH_SECONDS = 5.0

# Kept for callers of the Dubai name.
STATE_MAX_AGE_SECONDS = RESOLVED_RETENTION_SECONDS

OBSERVED, CANDIDATE, UNATTENDED, ALERTED, RESOLVED = "OBSERVED", "CANDIDATE", "UNATTENDED", "ALERTED", "RESOLVED"
REMOVED, MOVED, ATTENDED, EXPIRED = "removed", "moved", "attended", "expired"

_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: Episode ids carry the run, like track labels, so two inference runs never share one.
RUN_ID = os.getenv("TRACK_RUN_ID") or "".join(random.choice(_BASE36) for _ in range(4))


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _diag(bbox):
    x1, y1, x2, y2 = bbox
    return math.hypot(x2 - x1, y2 - y1)


def _area(bbox):
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _point_to_box_distance(px, py, box):
    """Shortest distance from a point to an axis-aligned box (0 if inside)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(object_tracks):
    """One track per physical object this frame: highest confidence wins."""
    kept = []
    for obj in sorted(object_tracks, key=lambda t: getattr(t, "confidence", 0.0) or 0.0, reverse=True):
        if all(_iou(obj.bbox, k.bbox) < DUPLICATE_IOU for k in kept):
            kept.append(obj)
    return kept


def _move_tolerance(*boxes):
    return max(MIN_MOVE_TOLERANCE_PX, MOVE_TOLERANCE_FRAC * max(_diag(b) for b in boxes))


def _similar_size(a, b):
    small, large = sorted((_area(a), _area(b)))
    return small > 0 and large / small <= ASSOCIATION_SIZE_RATIO


class EpisodeObject:
    """The episode's last sighting, shaped like a tracking.TrackedObject, for the event of a
    resolution - which by its nature usually happens while the object is NOT being detected."""

    group = "object"
    is_new = False
    raw_track_id = None

    def __init__(self, episode):
        self.camera_id = episode["camera_id"]
        self.track_id = episode["track_id"]
        self.class_id = episode["class_id"]
        self.class_name = episode["class_name"]
        self.confidence = episode["confidence"]
        self.bbox = [int(v) for v in episode["bbox"]]


class AbandonedObjectDetector:
    """Per-camera episode state machine over the object-group (bag/box) tracks.

    Gated on the camera's own abandoned_object feature flag, refreshed live via update_config - the
    same pattern as ZoneEvaluator / LineEvaluator, so a dashboard toggle takes effect without a
    restart. Every camera is independent: no episode is ever shared between cameras.
    """

    def __init__(self, camera_features, dwell_seconds=DWELL_SECONDS):
        self.dwell = dwell_seconds
        self._features = camera_features
        self._episodes = {}                  # episode_id -> episode dict (live + recently resolved)
        self._track_episode = {}             # (camera_id, track_id) -> episode_id
        self._resolutions = []               # resolved ALERTED episodes awaiting drain_resolutions()
        self._last_update = {}               # camera_id -> time of its previous update()
        self._counter = itertools.count(1)
        self._now = 0.0
        self._next_upkeep = None

    # ------------------------------------------------------------------ config
    def enabled_for(self, camera_id):
        return bool(self._features.get(camera_id, {}).get("abandoned_object"))

    def cameras_enabled(self):
        return [cid for cid in self._features if self.enabled_for(cid)]

    def update_config(self, camera_features):
        self._features = camera_features

    def forget_camera(self, camera_id):
        """Drop all state for a camera that stopped being inferred (its open alerts stay open in the
        dashboard: nothing was observed to resolve them)."""
        for eid in [e for e, ep in self._episodes.items() if ep["camera_id"] == camera_id]:
            del self._episodes[eid]
        for key in [k for k in self._track_episode if k[0] == camera_id]:
            del self._track_episode[key]
        self._resolutions = [r for r in self._resolutions if r[0].camera_id != camera_id]
        self._last_update.pop(camera_id, None)

    # ------------------------------------------------------------------ queries
    def state_for_track(self, camera_id, track_id):
        """The episode this track currently belongs to, or None."""
        return self._episodes.get(self._track_episode.get((camera_id, track_id)))

    def episodes(self, camera_id=None):
        """Snapshots of every live and recently resolved episode (reporting, tests)."""
        return [dict(ep, track_ids=list(ep["track_ids"])) for ep in self._episodes.values()
                if camera_id is None or ep["camera_id"] == camera_id]

    def active_count(self):
        """Objects currently on the unattended-dwell clock (not yet alerted) and seen within
        ACTIVE_FRESH_SECONDS of the latest update."""
        return sum(1 for ep in self._episodes.values()
                   if ep["state"] == UNATTENDED and self._now - ep["last_seen"] <= ACTIVE_FRESH_SECONDS)

    def alerted_count(self):
        """Objects currently in the abandoned (alerted, not yet resolved) state."""
        return sum(1 for ep in self._episodes.values() if ep["state"] == ALERTED)

    def drain_resolutions(self, camera_id=None):
        """[(EpisodeObject, metadata)] for ALERTED episodes resolved since the last drain - each one
        exactly once - so the caller can report the recovery of the alert it raised."""
        out = [r for r in self._resolutions if camera_id is None or r[0].camera_id == camera_id]
        self._resolutions = [r for r in self._resolutions if not (camera_id is None or r[0].camera_id == camera_id)]
        return out

    # ------------------------------------------------------------------ update
    def update(self, camera_id, object_tracks, person_boxes, frame_width, frame_height, now=None):
        """
        object_tracks : list of TrackedObject in the 'object' group
        person_boxes  : list of [x1, y1, x2, y2] person boxes seen this frame
        now           : the frame's time (seconds); time.monotonic() when omitted
        returns       : [(tracked_object, metadata)] for episodes that have JUST crossed the
                        abandonment threshold - at most once per episode. Resolutions of alerted
                        episodes are collected for drain_resolutions().

        Returns [] when the camera's abandoned_object flag is off.
        """
        if not self.enabled_for(camera_id):
            return []
        now = time.monotonic() if now is None else float(now)
        self._now = now
        previous = self._last_update.get(camera_id)
        frame_gap = min(max(now - previous, 0.0), MAX_FRAME_GAP_SECONDS) if previous is not None else 0.0
        self._last_update[camera_id] = now
        prox_radius = PROXIMITY_FRAC * (math.hypot(frame_width, frame_height) or 1.0)

        fired, claimed = [], {}
        candidates = [t for t in object_tracks if getattr(t, "class_id", None) in CANDIDATE_CLASS_IDS]
        for obj in _dedupe(candidates):
            cx, cy = _center(obj.bbox)
            episode = self._associate(camera_id, obj, cx, cy, now, claimed)
            if episode is None:
                continue                         # a fragment of an object already seen this frame
            claimed[episode["episode_id"]] = list(obj.bbox)
            episode, alert = self._observe(episode, obj, cx, cy, now, person_boxes, prox_radius)
            claimed[episode["episode_id"]] = list(obj.bbox)
            if alert is not None:
                fired.append(alert)

        # Episodes whose object was NOT detected this frame. People are still detected, so attendance
        # is still decidable at the resting position.
        for episode in [ep for ep in self._episodes.values()
                        if ep["camera_id"] == camera_id and ep["state"] != RESOLVED
                        and ep["episode_id"] not in claimed]:
            self._unseen(episode, now, frame_gap, person_boxes, prox_radius)

        if self._next_upkeep is None or now >= self._next_upkeep:
            self._next_upkeep = now + 30.0
            self.prune(now=now)
        return fired

    # ------------------------------------------------------------------ association
    def _associate(self, camera_id, obj, cx, cy, now, claimed):
        """The episode this detection belongs to: its track's, else an unseen nearby one of similar
        size and compatible class still inside the recovery window, else a new one. None when it is
        a fragment of an episode already seen this frame."""
        episode = self._episodes.get(self._track_episode.get((camera_id, obj.track_id)))
        if episode is not None and episode["state"] != RESOLVED and episode["episode_id"] not in claimed:
            return episode
        for box in claimed.values():
            if _iou(obj.bbox, box) >= FRAGMENT_IOU or _point_to_box_distance(cx, cy, box) == 0.0:
                return None
        best, best_distance = None, None
        for ep in self._episodes.values():
            if (ep["camera_id"] != camera_id or ep["state"] == RESOLVED or ep["episode_id"] in claimed
                    or now - ep["last_seen"] > RECOVERY_WINDOW_SECONDS):
                continue
            if ASSOCIATION_STRICT_CLASS and ep["class_id"] != obj.class_id:
                continue
            if not _similar_size(obj.bbox, ep["anchor_bbox"]):
                continue
            distance = math.hypot(cx - ep["anchor"][0], cy - ep["anchor"][1])
            tolerance = max(MIN_MOVE_TOLERANCE_PX,
                            ASSOCIATION_DISTANCE_FRAC * max(_diag(obj.bbox), _diag(ep["anchor_bbox"])))
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best, best_distance = ep, distance
        if best is not None:
            return best
        return self._new_episode(camera_id, obj, cx, cy, now)

    def _new_episode(self, camera_id, obj, cx, cy, now):
        episode_id = f"UA-{RUN_ID}-{next(self._counter):04d}"
        episode = {
            "episode_id": episode_id, "camera_id": camera_id, "state": OBSERVED, "resolution": None,
            "class_id": obj.class_id, "class_name": getattr(obj, "class_name", str(obj.class_id)),
            "class_votes": {}, "confidence": float(getattr(obj, "confidence", 0.0) or 0.0),
            "track_id": obj.track_id, "track_ids": [], "anchor": (cx, cy), "anchor_bbox": list(obj.bbox),
            "bbox": list(obj.bbox), "first_seen": now, "last_seen": now, "observed_seconds": 0.0,
            "grace_started_at": None, "unattended_since": None, "alerted_at": None, "resolved_at": None,
            "attended_since": None, "last_near": None, "move_since": None, "unseen_in_view": 0.0,
            "unseen_total": 0.0,
        }
        self._episodes[episode_id] = episode
        return episode

    # ------------------------------------------------------------------ one detected object
    def _observe(self, ep, obj, cx, cy, now, person_boxes, prox_radius):
        """Advance an episode on a frame its object was detected in. Returns (episode, alert|None);
        the episode differs from the one passed in when the object turned out to have moved."""
        if obj.track_id not in ep["track_ids"]:
            ep["track_ids"].append(obj.track_id)
        self._track_episode[(ep["camera_id"], obj.track_id)] = ep["episode_id"]
        previous_seen = ep["last_seen"]
        ep.update(track_id=obj.track_id, bbox=list(obj.bbox), last_seen=now, unseen_in_view=0.0, unseen_total=0.0,
                  confidence=float(getattr(obj, "confidence", 0.0) or 0.0))
        name = getattr(obj, "class_name", str(obj.class_id))
        ep["class_votes"][name] = ep["class_votes"].get(name, 0) + 1
        if ep["class_votes"][name] >= ep["class_votes"].get(ep["class_name"], 0):
            ep["class_id"], ep["class_name"] = obj.class_id, name

        near = self._attend(ep, now, person_boxes, prox_radius)

        # ---- moved? A resting object only moves when someone handles it: a drift past the tolerance
        # that PERSISTS, with a person at the spot around it, is a move. One bad box is not, nor is a
        # box that wanders with nobody there (it holds no dwell progress either - see below).
        off_spot = math.hypot(cx - ep["anchor"][0], cy - ep["anchor"][1]) > _move_tolerance(obj.bbox, ep["anchor_bbox"])
        if off_spot:
            if ep["move_since"] is None:
                ep["move_since"] = now
            handled = ep["last_near"] is not None and ep["last_near"] >= ep["move_since"] - MOVE_HANDLING_SECONDS
            if handled and now - ep["move_since"] >= MOVE_CONFIRM_SECONDS:
                self._resolve(ep, MOVED, now)
                moved_to = self._new_episode(ep["camera_id"], obj, cx, cy, now)
                return self._observe(moved_to, obj, cx, cy, now, person_boxes, prox_radius)
        else:
            ep["move_since"] = None

        if ep["state"] == ALERTED:
            self._resolve_if_attended(ep, now)
            return ep, None
        if off_spot:
            return ep, None                      # not at its resting spot (yet): no dwell progress

        # ---- before the alert: a person near the object holds its clock
        if near:
            if now - ep["attended_since"] >= PREALERT_ATTEND_SECONDS:
                self._reset(ep)
            return ep, None

        # ---- grace: absorb a brief missed-person detection before this counts as unattended
        if ep["grace_started_at"] is None:
            ep["grace_started_at"] = now
            ep["state"] = CANDIDATE
        else:
            ep["observed_seconds"] += min(max(now - previous_seen, 0.0), OBSERVATION_CREDIT_CAP_SECONDS)
        if now - ep["grace_started_at"] < GRACE_SECONDS:
            return ep, None

        # ---- stationary AND unattended past grace: the dwell clock runs from the END of grace (the
        # frames in between were watched for people, so they were unattended).
        if ep["unattended_since"] is None:
            ep["unattended_since"] = ep["grace_started_at"] + GRACE_SECONDS
            ep["state"] = UNATTENDED
        dwell = now - ep["unattended_since"]
        if dwell >= self.dwell and ep["observed_seconds"] >= MIN_OBSERVED_FRACTION * self.dwell:
            ep["state"], ep["alerted_at"] = ALERTED, now
            return ep, (obj, self._alert_metadata(ep, obj, dwell, cx, cy))
        return ep, None

    # ------------------------------------------------------------------ one undetected object
    def _unseen(self, ep, now, frame_gap, person_boxes, prox_radius):
        near = self._attend(ep, now, person_boxes, prox_radius)
        ep["unseen_total"] += frame_gap
        if not near:
            ep["unseen_in_view"] += frame_gap
        if ep["state"] == ALERTED:
            if self._resolve_if_attended(ep, now):
                return
        elif near and ep["grace_started_at"] is not None and now - ep["attended_since"] >= PREALERT_ATTEND_SECONDS:
            # an owner who comes back while the bag happens to be missed still holds the clock
            self._reset(ep)
        if ep["unseen_in_view"] >= RECOVERY_WINDOW_SECONDS or ep["unseen_total"] >= MAX_UNSEEN_SECONDS:
            self._resolve(ep, REMOVED if ep["state"] == ALERTED else EXPIRED, now)

    # ------------------------------------------------------------------ helpers
    def _attend(self, ep, now, person_boxes, prox_radius):
        """Is a person near the resting position now? Keeps the current attendance run: presence
        with gaps no longer than GRACE_SECONDS (a missed person detection) is one run."""
        near = REQUIRE_OWNERLESS and any(
            _point_to_box_distance(ep["anchor"][0], ep["anchor"][1], pb) <= prox_radius for pb in person_boxes)
        if near:
            if ep["attended_since"] is None or now - ep["last_near"] > GRACE_SECONDS:
                ep["attended_since"] = now
            ep["last_near"] = now
        return near

    def _resolve_if_attended(self, ep, now):
        if (ep["attended_since"] is not None and now - ep["last_near"] <= GRACE_SECONDS
                and ep["last_near"] - ep["attended_since"] >= ATTENDED_RESOLVE_SECONDS):
            self._resolve(ep, ATTENDED, now)
            return True
        return False

    @staticmethod
    def _reset(ep):
        ep.update(state=OBSERVED, unattended_since=None, grace_started_at=None, observed_seconds=0.0)

    def _resolve(self, ep, reason, now):
        alerted = ep["state"] == ALERTED
        ep.update(state=RESOLVED, resolution=reason, resolved_at=now)
        for track_id in ep["track_ids"]:
            if self._track_episode.get((ep["camera_id"], track_id)) == ep["episode_id"]:
                del self._track_episode[(ep["camera_id"], track_id)]
        if alerted:
            self._resolutions.append((EpisodeObject(ep), self._resolution_metadata(ep, reason, now)))

    def _episode_metadata(self, ep):
        return {"episode_id": ep["episode_id"], "track_ids": list(ep["track_ids"]),
                "track_changes": max(0, len(ep["track_ids"]) - 1), "object_type": ep["class_name"]}

    def _alert_metadata(self, ep, obj, dwell, cx, cy):
        return dict(self._episode_metadata(ep), episode_state="alerted", unattended=True,
                    dwell_seconds=round(dwell, 1), observed_seconds=round(ep["observed_seconds"], 1),
                    seen_for_seconds=round(ep["last_seen"] - ep["first_seen"], 1), position=[int(cx), int(cy)])

    def _resolution_metadata(self, ep, reason, now):
        return dict(self._episode_metadata(ep), episode_state="resolved", recovered=True, resolution=reason,
                    duration_seconds=round(now - ep["alerted_at"], 1),
                    unattended_seconds=round(now - (ep["unattended_since"] or ep["alerted_at"]), 1),
                    position=[int(ep["anchor"][0]), int(ep["anchor"][1])],
                    # a resolution is a state change, not a sighting: no clip
                    video={"required": False})

    # ------------------------------------------------------------------- upkeep
    def prune(self, max_age_seconds=RESOLVED_RETENTION_SECONDS, now=None):
        """Forget episodes resolved more than max_age_seconds ago. Called from update()."""
        now = self._now if now is None else now
        stale = [eid for eid, ep in self._episodes.items()
                 if ep["state"] == RESOLVED and now - ep["resolved_at"] > max_age_seconds]
        for eid in stale:
            del self._episodes[eid]
        for key in [k for k, eid in self._track_episode.items() if eid not in self._episodes]:
            del self._track_episode[key]
        return len(stale)
