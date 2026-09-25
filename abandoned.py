"""
Abandoned object detection - unattended bag/box alerts.  [RFP 6.1]

An object (backpack / handbag / suitcase) is ABANDONED when it has stayed
STATIONARY and UNATTENDED - no person nearby - continuously for longer than a
dwell threshold.

Why it needs tracking: "stationary" and "for N seconds" are only decidable
across frames, so this sits on top of the JeztSort object-group tracks (O-####)
exactly the way zones.py sits on top of person / vehicle tracks.

The two-condition rule is what separates a real left-behind bag from a bag its
owner is standing beside: while a person is within the proximity radius the
dwell clock is held at zero, so only an object whose owner has walked away and
STAYED away trips the alarm. It also catches an object that was already sitting
there when the camera came up, because the clock starts the first frame it is
seen stationary with nobody near it.

The alarm fires once per resting spot. If the object is picked up and moved the
state resets, so setting it down elsewhere starts a fresh dwell.

A person detector can miss a frame or two even while the owner is standing
right there - a turn, a brief occlusion. Going straight from "attended" to
"the dwell clock is running" on the very next frame would let that flicker
restart what should have been continuous attendance. ABANDON_OWNER_GRACE_SECONDS
absorbs a short window of apparent non-attendance before the object counts as
genuinely unattended at all; only once that elapses does the real dwell clock
(above) start. Grace time is real seconds, same FPS-independent basis as
DWELL_SECONDS, not a frame count - a frame count would give the grace window
a different real-world duration on every camera's actual processing rate.

STATE IS KEPT PER RESTING SPOT, NOT PER TRACK ID  (2026-09-15)
--------------------------------------------------------------
This used to key its clock on the track id, which made a real test on CAM-R25
impossible to trip. A black backpack left against a wall was detected at
0.26-0.58 confidence - either side of the camera's 0.45 floor - so it dropped
out of the detections for 2.5-10 s at a time. tracking.py gives a track that
has coasted for more than 1.5 s a NEW label when it is matched again (a rule
that exists to stop one vehicle's plate being attributed to the next), so every
dropout handed this module a brand-new track id, and a brand-new id started the
30 s dwell from zero. Replayed from the DVR, the longest the bag ever kept one
id was 15 s. It could never have fired, however long it was left.

A left bag does not move, so its POSITION is the identity that survives a
detector dropout. A track that appears within the move tolerance of a spot
seen in the last SPOT_MEMORY_SECONDS continues that spot's clock.

And because the clock may now run across frames where the bag itself is not
detected, "unattended" is judged against the spot on EVERY frame, including
those frames: people are detected independently of the bag, so an owner who
comes back while the bag happens to be missed still holds the clock at zero.
That is also what makes it safe for the dwell to start when the grace window
ENDS (grace_started_at + GRACE_SECONDS) rather than at whichever later frame
next happens to detect the bag - the gap has been watched for people too.

Env overrides:
  ABANDON_DWELL_SECONDS      unattended-stationary time before alert  (default 30)
  ABANDON_MOVE_TOLERANCE     max centre drift still counted as 'still',
                             as a fraction of the object's diagonal    (default 0.5)
  ABANDON_PROXIMITY          person-near radius, fraction of frame diag (default 0.12)
  ABANDON_REQUIRE_OWNERLESS  1 = require nobody near, 0 = pure stationary(default 1)
  ABANDON_OWNER_GRACE_SECONDS  grace before "no owner" starts the dwell
                                clock, absorbing a brief missed-person
                                detection                          (default 2.0)
  ABANDON_SPOT_MEMORY_SECONDS  how long a resting spot survives the object
                                not being detected                 (default 30)
  ABANDON_CLASS_IDS          COCO ids that can be abandoned  (default 24,26,28 =
                             backpack, handbag, suitcase)
  ABANDON_MIN_OBSERVED_FRACTION  share of the dwell the object must actually
                                be DETECTED before it can alarm       (default 0.3)
  ABANDON_OBJECT_CONFIDENCE  detector confidence a candidate bag needs to reach
                             this analytic, below the camera's own floor
                             (default 0.25) - see OBJECT_CONFIDENCE
"""
import math
import os
import time


DWELL_SECONDS = float(os.getenv("ABANDON_DWELL_SECONDS", "30"))
MOVE_TOLERANCE_FRAC = float(os.getenv("ABANDON_MOVE_TOLERANCE", "0.5"))
PROXIMITY_FRAC = float(os.getenv("ABANDON_PROXIMITY", "0.12"))
REQUIRE_OWNERLESS = os.getenv("ABANDON_REQUIRE_OWNERLESS", "1") == "1"
GRACE_SECONDS = float(os.getenv("ABANDON_OWNER_GRACE_SECONDS", "2.0"))

# How long a resting spot is remembered while its object is not detected.
# Matches tracking.py's 30 s object coast window: an object missed for longer
# than that has no track to come back to either.
SPOT_MEMORY_SECONDS = float(os.getenv("ABANDON_SPOT_MEMORY_SECONDS", "30"))

# COCO ids that can be abandoned. Handbag (26) was missing: the detector labels
# a carried backpack "handbag" as often as not, and most of the bags logged on
# CAM-R25 on 2026-09-15 were handbags - none of which could ever alarm.
CANDIDATE_CLASS_IDS = frozenset(
    int(v) for v in os.getenv("ABANDON_CLASS_IDS", "24,26,28").split(",") if v.strip())

# Confidence floor for candidate bags on a camera with abandoned_object armed.
# A still black bag against a wall scored 0.26-0.58 on CAM-R25, either side of
# the camera's 0.45 floor, and every dip was a dropout. multicam_inf.py lets
# candidate classes through at this floor into the tracker and this analytic
# only; the object-detection log keeps the camera's own floor. Replayed on the
# 2026-09-15 16:03-18:03 DVR: at 0.25 the test bag was observed 16.2 s of a 20 s
# dwell (13.1 s at 0.45) and nothing else alarmed.
OBJECT_CONFIDENCE = float(os.getenv("ABANDON_OBJECT_CONFIDENCE", "0.25"))

# Spot continuity lets the clock run across detector dropouts, so something must
# stop a spot that is barely ever seen from alarming: two brief false detections
# at the same place 29 s apart would otherwise add up to "30 s unattended". Each
# observation credits the time since the previous one, capped at
# OBSERVATION_CREDIT_CAP_SECONDS, and the alarm needs this share of the dwell.
MIN_OBSERVED_FRACTION = float(os.getenv("ABANDON_MIN_OBSERVED_FRACTION", "0.3"))
OBSERVATION_CREDIT_CAP_SECONDS = 0.5

# Minimum centre drift (px) that still counts as 'still' regardless of box size,
# so detection jitter on a small far-away box does not reset the dwell.
MIN_MOVE_TOLERANCE_PX = 15.0

# A spot's state is forgotten this long after its object was last seen.
STATE_MAX_AGE_SECONDS = 120.0

# active_count() reports a clock as running only if its object was seen this
# recently. State is deliberately kept long after an object leaves (above), and
# counting it as "on clock" for that whole time is what made a stopped clock
# read as a running one for two minutes on the status line.
ACTIVE_FRESH_SECONDS = 5.0


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _diag(bbox):
    x1, y1, x2, y2 = bbox
    return math.hypot(x2 - x1, y2 - y1)


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
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


#: Two candidate boxes overlapping this much in one frame are one physical bag.
#: At low confidence the detector reports a bag as "backpack" AND "handbag" with
#: near-identical boxes; the tracker keeps them as two tracks, and without this
#: one left bag raised two alarms (seen in the CAM-R25 replay at 0.25).
DUPLICATE_IOU = 0.5


def _dedupe(object_tracks):
    """One track per physical object this frame: highest confidence wins."""
    kept = []
    for obj in sorted(object_tracks, key=lambda t: getattr(t, "confidence", 0.0) or 0.0, reverse=True):
        if all(_iou(obj.bbox, k.bbox) < DUPLICATE_IOU for k in kept):
            kept.append(obj)
    return kept


def _move_tolerance(bbox):
    return max(MIN_MOVE_TOLERANCE_PX, MOVE_TOLERANCE_FRAC * _diag(bbox))


class AbandonedObjectDetector:
    """Per-camera state machine over the object-group (bag/box) tracks.

    Gated on the camera's own abandoned_object feature flag, refreshed live via
    update_config - the same pattern as ZoneEvaluator / LineEvaluator, so a
    dashboard toggle takes effect without a restart.
    """

    def __init__(self, camera_features, dwell_seconds=DWELL_SECONDS):
        self.dwell = dwell_seconds
        # camera_id -> feature flag dict
        self._features = camera_features
        # (camera_id, spot_id) -> state dict
        self._state = {}
        # (camera_id, track_id) -> spot_id
        self._track_spot = {}
        self._next_spot = 0

    # ------------------------------------------------------------------ config
    def enabled_for(self, camera_id):
        return bool(self._features.get(camera_id, {}).get("abandoned_object"))

    def cameras_enabled(self):
        return [cid for cid in self._features if self.enabled_for(cid)]

    def update_config(self, camera_features):
        self._features = camera_features

    def forget_camera(self, camera_id):
        """Drop all state for a camera that stopped being inferred."""
        for key in [k for k in self._state if k[0] == camera_id]:
            del self._state[key]
        for key in [k for k in self._track_spot if k[0] == camera_id]:
            del self._track_spot[key]

    def state_for_track(self, camera_id, track_id):
        """The state record currently driving this track, or None."""
        spot = self._track_spot.get((camera_id, track_id))
        return None if spot is None else self._state.get((camera_id, spot))

    # ------------------------------------------------------------------ update
    def _spot_for(self, camera_id, obj, cx, cy, now, claimed):
        """The resting spot this track belongs to: the one it is already bound
        to, else a recently seen unclaimed spot within move tolerance of it,
        else a new one."""
        spot = self._track_spot.get((camera_id, obj.track_id))
        if spot is not None and (camera_id, spot) in self._state and spot not in claimed:
            return spot

        tol = _move_tolerance(obj.bbox)
        best, best_distance = None, None
        for (cid, candidate), state in self._state.items():
            if cid != camera_id or candidate in claimed:
                continue
            if now - state["last_seen"] > SPOT_MEMORY_SECONDS:
                continue
            distance = math.hypot(cx - state["anchor"][0], cy - state["anchor"][1])
            if distance <= tol and (best_distance is None or distance < best_distance):
                best, best_distance = candidate, distance

        if best is None:
            self._next_spot += 1
            best = self._next_spot
            self._state[(camera_id, best)] = {
                "anchor": (cx, cy), "unattended_since": None,
                "grace_started_at": None, "alerted": False, "last_seen": now,
                "observed_seconds": 0.0,
            }

        self._track_spot[(camera_id, obj.track_id)] = best
        return best

    @staticmethod
    def _reset(state):
        state["unattended_since"] = None
        state["grace_started_at"] = None
        state["alerted"] = False
        state["observed_seconds"] = 0.0

    def _attended(self, cx, cy, person_boxes, prox_radius):
        if not REQUIRE_OWNERLESS:
            return False
        return any(_point_to_box_distance(cx, cy, pb) <= prox_radius
                   for pb in person_boxes)

    def update(self, camera_id, object_tracks, person_boxes,
               frame_width, frame_height):
        """
        object_tracks : list of TrackedObject in the 'object' group
        person_boxes  : list of [x1, y1, x2, y2] person boxes seen this frame
        returns       : list of (tracked_object, metadata) for objects that have
                        JUST crossed the abandonment threshold (fires once each)

        Returns [] when the camera's abandoned_object flag is off.
        """
        if not self.enabled_for(camera_id):
            return []

        now = time.monotonic()
        fired = []

        frame_diag = math.hypot(frame_width, frame_height) or 1.0
        prox_radius = PROXIMITY_FRAC * frame_diag

        claimed = set()

        for obj in _dedupe(object_tracks):

            cx, cy = _center(obj.bbox)
            spot = self._spot_for(camera_id, obj, cx, cy, now, claimed)
            claimed.add(spot)
            state = self._state[(camera_id, spot)]
            previous_seen = state["last_seen"]
            state["last_seen"] = now

            # ---- moved? this is not the same resting spot - reset everything --
            moved = math.hypot(cx - state["anchor"][0],
                               cy - state["anchor"][1]) > _move_tolerance(obj.bbox)
            if moved:
                state["anchor"] = (cx, cy)
                self._reset(state)
                continue

            # ---- attended? a person within the radius holds the dwell clock ---
            if self._attended(cx, cy, person_boxes, prox_radius):
                self._reset(state)
                continue

            # ---- grace: absorb a brief missed-person detection before this
            # counts as genuinely unattended at all - see module docstring.
            if state["grace_started_at"] is None:
                state["grace_started_at"] = now
            else:
                state["observed_seconds"] += min(max(now - previous_seen, 0.0),
                                                 OBSERVATION_CREDIT_CAP_SECONDS)

            if now - state["grace_started_at"] < GRACE_SECONDS:
                continue

            # ---- stationary AND unattended past grace: run the dwell clock ---
            # From the moment grace ENDED, not from this frame: the frames in
            # between were watched for people (below), so they were unattended.
            if state["unattended_since"] is None:
                state["unattended_since"] = state["grace_started_at"] + GRACE_SECONDS

            dwell = now - state["unattended_since"]

            if (dwell >= self.dwell and not state["alerted"]
                    and state["observed_seconds"] >= MIN_OBSERVED_FRACTION * self.dwell):
                state["alerted"] = True
                fired.append((obj, {
                    "object_type": obj.class_name,
                    "dwell_seconds": round(dwell, 1),
                    "observed_seconds": round(state["observed_seconds"], 1),
                    "unattended": True,
                    "position": [int(cx), int(cy)],
                }))

        # ---- spots whose object was NOT detected this frame -------------------
        # People are still detected, so attendance is still decidable at the
        # spot's anchor: an owner who returns while the bag is missed resets it.
        for (cid, spot), state in self._state.items():
            if cid != camera_id or spot in claimed:
                continue
            if now - state["last_seen"] > SPOT_MEMORY_SECONDS:
                continue
            if (state["grace_started_at"] is not None or state["alerted"]) and \
                    self._attended(*state["anchor"], person_boxes, prox_radius):
                self._reset(state)

        return fired

    # ------------------------------------------------------------------- upkeep
    def prune(self, max_age_seconds=STATE_MAX_AGE_SECONDS):
        """Forget spots not seen for a while (object left, id retired)."""
        cutoff = time.monotonic() - max_age_seconds
        stale = [k for k, s in self._state.items()
                 if s.get("last_seen", 0.0) < cutoff]
        for k in stale:
            del self._state[k]
        live = set(self._state)
        for key in [k for k, spot in self._track_spot.items()
                    if (k[0], spot) not in live]:
            del self._track_spot[key]
        return len(stale)

    def active_count(self):
        """Objects currently on the unattended-dwell clock (not yet fired)
        and seen within ACTIVE_FRESH_SECONDS."""
        cutoff = time.monotonic() - ACTIVE_FRESH_SECONDS
        return sum(1 for s in self._state.values()
                   if s.get("unattended_since") and not s.get("alerted")
                   and s.get("last_seen", 0.0) >= cutoff)

    def alerted_count(self):
        """Objects currently in the abandoned (alarmed) state."""
        return sum(1 for s in self._state.values() if s.get("alerted"))
