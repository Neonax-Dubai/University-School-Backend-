"""
Zone geometry - intrusion, perimeter breach and loitering.

Zones come from the dashboard (GET /api/ai/cameras/) as polygons in NORMALISED
coordinates, 0.0-1.0 of frame width/height. Normalised means the same zone
works whatever resolution the stream decodes at, so zones stay valid whether a
camera serves 1920x1080 (analytics now run on the main stream) or any other size.

A tracked object is reduced to one anchor point, normalised the same way, and
tested against each polygon with a standard ray-casting point-in-polygon test.

  ZONE_ANCHOR=center   bbox centre                     (default)
  ZONE_ANCHOR=bottom   bottom-centre, i.e. the feet

`center` is the default because it is what the integration brief specifies.
`bottom` is usually the more accurate choice for zones drawn on the floor or
ground: a person's centre sits at torso height, so a centre anchor reports them
inside a ground zone while they are still short of it, and keeps reporting them
inside after they step out. Switch with the env var if zones read early.

Four per-object rules are implemented, and each is gated twice - the camera's
feature flag must be on AND the zone's own rule must be on:

  intrusion          while the object is inside the zone   (debounced)
  perimeter_breach   only on the outside -> inside crossing (edge triggered)
  loitering          once, after a continuous dwell exceeds the threshold
  vehicle_detection  a VEHICLE entering a zone designated for vehicles
                     (edge triggered, no dwell - see evaluate())

Env overrides:
  ZONE_ANCHOR                       center | bottom          (default center)
  LOITERING_THRESHOLD_SECONDS       dwell before loitering   (default 10)
  CROWD_DROP_GRACE_SECONDS          flicker grace before re-arm (default 1.0)
  VEHICLE_ZONE_EXIT_GRACE_SECONDS   time outside before a vehicle counts as
                                    having arrived again    (default 5.0)
"""

import os
import time

from tracking import VEHICLE_GROUP


ANCHOR = os.getenv("ZONE_ANCHOR", "center").lower()

# The dashboard event type raised by the zone rule "vehicle_detection".
# Deliberately NOT the generic vehicle_detected: that one is presence sampling
# for every vehicle anywhere in frame, and overloading it would make "a vehicle
# reached the ANPR gate" indistinguishable from "a van is parked across the
# road". Kept as a constant so the string exists in exactly one place on the
# inference side.
VEHICLE_ZONE_EVENT = "vehicle_zone_detection"

# How long a vehicle must be continuously OUTSIDE before returning counts as a
# fresh arrival rather than the same one.
#
# Measured on CAM-R06 (an ANPR entry ramp): vehicles come to rest with their
# anchor sitting almost exactly on the zone edge, and the tracked box breathes
# by a pixel or two per frame. That crosses the polygon boundary repeatedly
# without the vehicle moving at all, and each crossing looked like a new
# arrival - one car produced four events in five minutes.
#
# A time-since-last-event cooldown cannot fix that: a vehicle parked on the
# edge simply re-fires every time the cooldown expires. What distinguishes
# jitter from departure is how long the vehicle stays out, which is what this
# measures. Same idea as CROWD_DROP_GRACE_SECONDS below, for the same reason.
#
#   TOO LOW  -> back to duplicate arrivals for one stationary vehicle.
#   TOO HIGH -> a car that pulls away from the gate and the next car that
#               pulls up behind it merge into one arrival, and the second
#               plate is never captured.
VEHICLE_ZONE_EXIT_GRACE_SECONDS = float(
    os.getenv("VEHICLE_ZONE_EXIT_GRACE_SECONDS", "5.0")
)
# Fallback only. Each zone now carries its own dwell in its dashboard payload
# ("loitering": {"dwell_seconds": N}); this applies when a zone predates that
# field or the dashboard is unreachable and a cached config is in use.
LOITERING_THRESHOLD_SECONDS = float(os.getenv("LOITERING_THRESHOLD_SECONDS", "10"))


def loitering_dwell_for(zone):
    """Dwell threshold for one zone, falling back to the module default."""
    config = zone.get("loitering") or {}
    try:
        dwell = float(config.get("dwell_seconds"))
    except (TypeError, ValueError):
        return LOITERING_THRESHOLD_SECONDS

    # A zero or negative dwell would make loitering fire instantly, which is
    # never what an operator means - treat it as unset.
    return dwell if dwell > 0 else LOITERING_THRESHOLD_SECONDS

# evaluate_crowd() re-arms (resets the hold timer) once the in-zone count
# drops below the threshold - but a detector/tracker flicker (one frame where
# a confirmed person's box is briefly missed) makes the count dip for a
# single frame even though everyone is still physically there, wiping out
# hold-time progress that may have been building for seconds. The count must
# stay continuously below threshold for this long before it is treated as a
# genuine dispersal rather than noise; a dip shorter than this is absorbed
# and the hold timer keeps running as if the count never dropped.
#   TOO LOW  -> back to the original problem: brief flicker resets the timer,
#               so a crowd that is genuinely holding rarely gets to fire.
#   TOO HIGH -> a crowd that truly DOES disperse takes this long to re-arm,
#               delaying the next detection of a NEW crowd forming in the
#               same zone.
# Time-based, not frame-count-based, so it behaves the same regardless of
# inference rate - test_crowd.py runs uncapped while multicam_inf.py runs at
# a fixed TARGET_INFERENCE_FPS, and a frame-count grace would silently mean a
# different real-world grace window in each.
CROWD_DROP_GRACE_SECONDS = float(os.getenv("CROWD_DROP_GRACE_SECONDS", "1.0"))

# State for a track/zone pair is forgotten this long after it was last seen.
STATE_MAX_AGE_SECONDS = 300.0


# ============================================================
# GEOMETRY
# ============================================================

def point_in_polygon(x, y, polygon):
    """Ray casting. polygon is [[x, y], ...] in the same units as x, y."""
    if not polygon or len(polygon) < 3:
        return False

    inside = False

    count = len(polygon)
    j = count - 1

    for i in range(count):

        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]

        # Does the edge straddle the horizontal ray through y?
        if (yi > y) != (yj > y):

            # x of the edge at height y. The straddle test guarantees yj != yi.
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi

            if x < crossing_x:
                inside = not inside

        j = i

    return inside


def anchor_point(bbox, frame_width, frame_height):
    """Reduce a pixel bbox to one normalised (x, y) point in 0.0-1.0."""
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0

    cy = float(y2) if ANCHOR == "bottom" else (y1 + y2) / 2.0

    if frame_width <= 0 or frame_height <= 0:
        return None

    return cx / frame_width, cy / frame_height


# ============================================================
# EVALUATOR
# ============================================================

class ZoneEvaluator:
    """
    Turns "this track is at this point" into zone events.

    Holds one state record per (camera, track, zone) so that edge-triggered
    rules (perimeter breach) and dwell rules (loitering) can be evaluated -
    neither is decidable from a single frame.
    """

    def __init__(self, camera_zones, camera_features):
        # camera_id -> [zone dict], only enabled zones
        self._zones = {
            camera_id: [z for z in zones if z.get("enabled", True)]
            for camera_id, zones in camera_zones.items()
        }

        # camera_id -> feature flag dict
        self._features = camera_features

        # (camera_id, track_id, zone_id) -> state
        self._state = {}

        # (camera_id, zone_id) -> crowd state
        self._crowd = {}

    # ------------------------------------------------------------------ info
    def zone_count(self):
        return sum(len(zones) for zones in self._zones.values())

    def cameras_with_zones(self):
        return [camera_id for camera_id, zones in self._zones.items() if zones]

    # -------------------------------------------------------------- internal
    def _rule_on(self, camera_id, zone, rule):
        """A rule fires only if the camera feature AND the zone rule are on."""
        if not self._features.get(camera_id, {}).get(rule):
            return False

        return bool(zone.get("rules", {}).get(rule))

    def _zone_metadata(self, zone):
        return {
            "zone_id": zone.get("id"),
            "zone_name": zone.get("name"),
            "zone_type": zone.get("type"),
        }

    # ------------------------------------------------------------------ eval
    def evaluate(self, tracked, frame_width, frame_height):
        """
        Returns a list of (event_type, metadata) for one tracked object.

        Must be called for every tracked object on every frame, including
        objects outside every zone - that is what makes the inside/outside
        transition observable.
        """
        zones = self._zones.get(tracked.camera_id)

        if not zones:
            return []

        point = anchor_point(tracked.bbox, frame_width, frame_height)

        if point is None:
            return []

        nx, ny = point

        now = time.monotonic()

        fired = []

        for zone in zones:

            zone_id = zone.get("id")

            key = (tracked.camera_id, tracked.track_id, zone_id)

            state = self._state.get(key)

            if state is None:
                state = {"inside": False, "entered_at": None, "loiter_fired": False}
                self._state[key] = state

            state["last_seen"] = now

            was_inside = state["inside"]

            is_inside = point_in_polygon(nx, ny, zone.get("coordinates", []))

            state["inside"] = is_inside

            # ---- left the zone: reset the dwell so re-entry starts fresh ----
            if not is_inside:
                state["entered_at"] = None
                state["loiter_fired"] = False

                # Only the vehicle rule reads this. Recorded on the transition
                # rather than every frame, so it is the moment of departure and
                # not the last frame seen.
                if was_inside:
                    state["left_at"] = now

                continue

            # ---- just crossed in ----
            if not was_inside:

                state["entered_at"] = now
                state["loiter_fired"] = False

                # Perimeter breach is the crossing itself, not the presence.
                if self._rule_on(tracked.camera_id, zone, "perimeter_breach"):

                    metadata = self._zone_metadata(zone)
                    metadata["transition"] = "outside_to_inside"

                    fired.append(("perimeter_breach", metadata))

                # Vehicle in a designated zone - the foundation for ANPR.
                #
                # ENTRY TRIGGERED, like perimeter_breach and for the same
                # reason: it answers "a vehicle arrived here", which is a
                # transition, not a state. Sitting inside this block means it
                # is structurally impossible to re-fire while the vehicle
                # remains in the zone, however long it stays - so this needs no
                # dwell timer and gets none. It is NOT loitering and must never
                # read the loitering dwell.
                #
                # The group test is what separates this from intrusion: those
                # rules fire for any tracked object, and a zone armed for
                # vehicles must ignore the people walking through it.
                # left_at is None on a track's first ever entry, which is a
                # genuine arrival and must not be held back.
                left_at = state.get("left_at")

                settled = (
                    left_at is None
                    or (now - left_at) >= VEHICLE_ZONE_EXIT_GRACE_SECONDS
                )

                if (
                    settled
                    and tracked.group == VEHICLE_GROUP
                    and self._rule_on(tracked.camera_id, zone, "vehicle_detection")
                ):
                    metadata = self._zone_metadata(zone)
                    metadata["transition"] = "outside_to_inside"

                    # class_name is already carried as metadata["object_type"]
                    # by build_track_event, so the vehicle class reaches ANPR
                    # without a second copy of the same string.
                    fired.append((VEHICLE_ZONE_EVENT, metadata))

            # ---- present inside the zone ----
            if self._rule_on(tracked.camera_id, zone, "intrusion"):
                fired.append(("intrusion", self._zone_metadata(zone)))

            # ---- dwell ----
            if (
                not state["loiter_fired"]
                and state["entered_at"] is not None
                and self._rule_on(tracked.camera_id, zone, "loitering")
            ):
                dwell = now - state["entered_at"]

                if dwell >= loitering_dwell_for(zone):

                    state["loiter_fired"] = True

                    metadata = self._zone_metadata(zone)
                    metadata["dwell_seconds"] = round(dwell, 1)
                    metadata["dwell_threshold"] = loitering_dwell_for(zone)

                    fired.append(("loitering", metadata))

        return fired

    # ------------------------------------------------------------------ crowd
    def evaluate_crowd(self, camera_id, tracked_objects, frame_width, frame_height):
        """
        Zone occupancy, evaluated once per frame rather than per object.

        Counts CONFIRMED PERSON TRACKS whose anchor falls inside the polygon.
        Counting tracks rather than raw YOLO boxes is what stops one person
        being counted twice when the detector splits them across frames, and it
        is why this reuses the tracker instead of a second model.

        The threshold must be held continuously for the zone's configured
        duration before the event fires, and it fires ONCE per crowd - it
        re-arms only after the count drops back below the threshold.
        """
        zones = self._zones.get(camera_id)

        if not zones:
            return []

        now = time.monotonic()

        fired = []

        for zone in zones:

            if not self._rule_on(camera_id, zone, "crowd_detection"):
                continue

            zone_id = zone.get("id")
            polygon = zone.get("coordinates", [])

            crowd_config = zone.get("crowd") or {}
            threshold = int(crowd_config.get("threshold", 10) or 10)
            duration = float(crowd_config.get("duration_seconds", 5) or 5)

            inside = []

            for tracked in tracked_objects:

                if tracked.group != "person":
                    continue

                point = anchor_point(tracked.bbox, frame_width, frame_height)

                if point is None:
                    continue

                if point_in_polygon(point[0], point[1], polygon):
                    inside.append(tracked)

            # Unique confirmed track ids - the whole point of counting tracks.
            count = len({tracked.track_id for tracked in inside})

            key = (camera_id, zone_id)

            state = self._crowd.get(key)

            if state is None:
                state = {"since": None, "fired": False, "below_since": None}
                self._crowd[key] = state

            state["last_seen"] = now
            state["count"] = count

            if count >= threshold:
                # Above threshold again - clear any grace window a prior
                # flicker started, so the NEXT dip gets its own full grace.
                state["below_since"] = None
            else:
                if state["below_since"] is None:
                    state["below_since"] = now

                if now - state["below_since"] >= CROWD_DROP_GRACE_SECONDS:
                    # Below threshold continuously for the full grace window -
                    # a genuine dispersal, not a flicker. Re-arm.
                    state["since"] = None
                    state["fired"] = False
                    state["below_since"] = None
                    continue
                # Still inside the grace window - treat this dip as detector/
                # tracker noise: the HOLD TIMER keeps running unbroken (a
                # missed frame must not wipe out seconds of real standing
                # time). That is a separate question from whether THIS tick
                # may fire - see the count < threshold guard below.

            if state["since"] is None:
                state["since"] = now

            held = now - state["since"]

            # count < threshold here means this tick is a grace-tolerated dip
            # (the branch above chose not to reset, but did not make the dip
            # disappear either). Firing must still wait for a tick where the
            # count itself genuinely meets the threshold again, or the event
            # would report a count that fails the very comparison it prints -
            # "count=9 >= 10" - even though the crowd really did hold near
            # enough to it. The hold timer above is what makes that NEXT
            # qualifying tick fire immediately instead of waiting a fresh
            # `duration` - only the report was ever wrong, not the timing.
            if state["fired"] or held < duration or count < threshold:
                continue

            state["fired"] = True

            metadata = self._zone_metadata(zone)
            metadata.update({
                "person_count": count,
                "threshold": threshold,
                "duration_seconds": round(held, 1),
            })

            # Box enclosing the crowd, so the evidence crop shows the group
            # rather than an arbitrary individual.
            xs1 = min(t.bbox[0] for t in inside)
            ys1 = min(t.bbox[1] for t in inside)
            xs2 = max(t.bbox[2] for t in inside)
            ys2 = max(t.bbox[3] for t in inside)

            # ...and the people the count was made of, so the still can show
            # WHICH ones. Same tracks, deduplicated the same way the count is
            # (unique track id), in a stable order. This changes no counting:
            # `count` above is still len({track ids}) and is computed before
            # this line. The boxes travel to the evidence overlay only - they
            # are not added to the event payload.
            contributors = []
            seen_ids = set()
            for tracked in inside:
                if tracked.track_id in seen_ids:
                    continue
                seen_ids.add(tracked.track_id)
                contributors.append(tracked)

            fired.append(("crowd_detected", metadata, [xs1, ys1, xs2, ys2],
                          contributors))

        return fired

    def crowd_counts(self):
        """Current occupancy per zone, for the status line."""
        return {
            key: state.get("count", 0)
            for key, state in self._crowd.items()
        }

    # ---------------------------------------------------------------- reload
    def update_config(self, camera_zones, camera_features):
        """
        Swap in freshly fetched zones and feature flags.

        Track state is kept for zones that still exist, so someone already
        standing inside a zone is not re-reported as a fresh crossing just
        because an unrelated camera was edited. State for deleted zones goes.

        Crowd dwell state is the one exception: it is also reset whenever an
        EXISTING zone's own crowd threshold/duration changes, not only when
        the zone disappears. Without this, a crowd already accumulating dwell
        time under an OLD (lower) threshold can outrun evaluate_crowd()'s
        CROWD_DROP_GRACE_SECONDS flicker-absorption window - the drop below
        the NEW threshold is still within its 1s grace when the dwell timer,
        started under the old rule, crosses `duration` - so crowd_detected
        fires once reporting the operator's NEW threshold in its metadata
        while the count that triggered it only ever satisfied the OLD one.
        """
        old_zones_by_key = {
            (camera_id, zone.get("id")): zone
            for camera_id, zones in self._zones.items()
            for zone in zones
        }

        self._zones = {
            camera_id: [z for z in zones if z.get("enabled", True)]
            for camera_id, zones in camera_zones.items()
        }

        self._features = camera_features

        live = {
            (camera_id, zone.get("id"))
            for camera_id, zones in self._zones.items()
            for zone in zones
        }

        stale = [key for key in self._state if (key[0], key[2]) not in live]

        for key in stale:
            del self._state[key]

        for key in [k for k in self._crowd if k not in live]:
            del self._crowd[key]

        for camera_id, zones in self._zones.items():
            for zone in zones:
                key = (camera_id, zone.get("id"))
                old_zone = old_zones_by_key.get(key)

                if old_zone is None:
                    continue

                if (old_zone.get("crowd") or {}) != (zone.get("crowd") or {}):
                    self._crowd.pop(key, None)

        return len(stale)

    # ----------------------------------------------------------------- prune
    def prune(self, max_age_seconds=STATE_MAX_AGE_SECONDS):
        cutoff = time.monotonic() - max_age_seconds

        stale = [
            key
            for key, state in self._state.items()
            if state.get("last_seen", 0) < cutoff
        ]

        for key in stale:
            del self._state[key]

        return len(stale)

    def tracked_keys(self):
        return len(self._state)
