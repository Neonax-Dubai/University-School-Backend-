"""
Line crossing - virtual tripwires on the confirmed track trajectory.

A crossing is a property of MOVEMENT, so it cannot be decided from one frame or
one YOLO detection. This works on the same stable track ids the rest of the
pipeline uses: the previous anchor point of a track and its current anchor form
a movement segment, and a crossing is reported when that segment genuinely
intersects the tripwire segment.

Testing only which SIDE of the line the track is on would be wrong - a track
far off the end of the line still changes side relative to the line's infinite
extension. Both segments are tested, so walking past the end of a tripwire does
not trigger it.

Direction convention (documented on the CameraLine model too):

    the line runs A(x1,y1) -> B(x2,y2)
    stand at A looking toward B, viewing the camera image

        "in"  = crossing left-to-right across the image
        "out" = crossing right-to-left across the image

    For a horizontal tripwire drawn left-to-right, "in" is therefore movement
    DOWN the image and "out" is movement UP it.

Swapping the endpoints swaps in and out, which is how a tripwire is re-aimed.

Env overrides:
  LINE_MIN_TRAVEL      ignore jitter below this normalised distance (default 0.005)
  LINE_MIN_POST_TRAVEL required travel AFTER crossing before the event fires
                       (default 0.02 = 2% of frame width; raise to suppress
                       triggers from a person whose centre only grazes the line)
"""

import os
import time


# A track that barely moves should not trip a line it happens to sit on.
MIN_TRAVEL = float(os.getenv("LINE_MIN_TRAVEL", "0.005"))

# How far (normalised) the track must travel PAST the line before the event
# fires.  Default 0.02 = 2 % of frame width on a 1920-px camera = ~38 px.
# Set to 0.0 to restore the old instant-trigger behaviour.
MIN_POST_TRAVEL = float(os.getenv("LINE_MIN_POST_TRAVEL", "0.02"))

STATE_MAX_AGE_SECONDS = 300.0

DIRECTION_IN = "in"
DIRECTION_OUT = "out"


# ============================================================
# GEOMETRY
# ============================================================

def side_of(line, px, py):
    """
    Signed area of the triangle (A, B, P).

    NOTE these are IMAGE coordinates, so y grows DOWNWARD. That flips the
    handedness relative to the usual maths convention, hence:

    > 0 : P is visually RIGHT of A->B (as seen on screen)
    < 0 : P is visually LEFT of A->B
    = 0 : P is on the line
    """
    return (
        (line["x2"] - line["x1"]) * (py - line["y1"])
        - (line["y2"] - line["y1"]) * (px - line["x1"])
    )


def _orientation(ax, ay, bx, by, cx, cy):
    value = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """
    True if segment AB properly intersects segment CD.

    The collinear/touching cases are deliberately treated as NOT crossing: a
    track that grazes the tripwire without passing through it should not raise
    an event, and the next frame will resolve it cleanly one way or the other.
    """
    d1 = _orientation(cx, cy, dx, dy, ax, ay)
    d2 = _orientation(cx, cy, dx, dy, bx, by)
    d3 = _orientation(ax, ay, bx, by, cx, cy)
    d4 = _orientation(ax, ay, bx, by, dx, dy)

    return d1 != d2 and d3 != d4 and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0


def anchor_point(bbox, frame_width, frame_height, anchor):
    """Normalised point representing the object, matching the zone anchor."""
    x1, y1, x2, y2 = bbox

    if frame_width <= 0 or frame_height <= 0:
        return None

    cx = (x1 + x2) / 2.0
    cy = float(y2) if anchor == "bottom" else (y1 + y2) / 2.0

    return cx / frame_width, cy / frame_height


# ============================================================
# EVALUATOR
# ============================================================

class LineEvaluator:
    """
    Holds the tripwires per camera and the last known position of each track,
    which is the only state a crossing test needs.
    """

    def __init__(self, camera_lines, camera_features, anchor="center"):
        self._lines = {
            camera_id: [ln for ln in lines if ln.get("enabled", True)]
            for camera_id, lines in camera_lines.items()
        }
        self._features = camera_features
        self._anchor = anchor

        # (camera_id, track_id) -> {"point": (x, y), "last_seen": monotonic}
        self._previous = {}

        # Pending crossings waiting for MIN_POST_TRAVEL confirmation.
        # (camera_id, track_id, line_id) -> {
        #     "event": (event_type, metadata),
        #     "cross_point": (x, y),   # normalised anchor at the moment of crossing
        #     "direction": "in"|"out",
        #     "last_seen": monotonic,
        # }
        self._pending = {}

    # ------------------------------------------------------------------ info
    def line_count(self):
        return sum(len(lines) for lines in self._lines.values())

    def cameras_with_lines(self):
        return [camera_id for camera_id, lines in self._lines.items() if lines]

    def update_config(self, camera_lines, camera_features, anchor=None):
        self._lines = {
            camera_id: [ln for ln in lines if ln.get("enabled", True)]
            for camera_id, lines in camera_lines.items()
        }
        self._features = camera_features

        if anchor is not None:
            self._anchor = anchor

    def pending_count(self):
        """Number of crossings waiting for MIN_POST_TRAVEL confirmation."""
        return len(self._pending)

    # ------------------------------------------------------------------ eval
    def evaluate(self, tracked, frame_width, frame_height):
        """
        Returns a list of (event_type, metadata) - normally empty.

        Must be called for every confirmed track on every frame, because the
        previous position it stores is what makes the next frame decidable.

        Two-phase crossing with MIN_POST_TRAVEL confirmation
        -----------------------------------------------------
        Phase 1  (segment intersection):
            The movement segment prev→current crosses the tripwire segment.
            The crossing is stored in _pending; nothing is returned yet.

        Phase 2  (post-crossing travel):
            On every subsequent frame, the distance from the stored crossing
            point to the current anchor is measured. When it exceeds
            MIN_POST_TRAVEL the event is emitted and the pending entry removed.

        If MIN_POST_TRAVEL is 0.0 the event fires immediately (old behaviour).
        """
        camera_id = tracked.camera_id

        point = anchor_point(tracked.bbox, frame_width, frame_height, self._anchor)

        if point is None:
            return []

        key = (camera_id, tracked.track_id)
        now = time.monotonic()

        record = self._previous.get(key)
        previous_point = record["point"] if record else None
        self._previous[key] = {"point": point, "last_seen": now}

        lines = self._lines.get(camera_id)

        # First sighting: nothing to compare against yet.
        if not lines or previous_point is None:
            return []

        # The camera-level feature switch gates every tripwire on it.
        if not self._features.get(camera_id, {}).get("line_crossing"):
            return []

        px, py = previous_point
        cx, cy = point

        fired = []

        # ------------------------------------------------------------------
        # Phase 2: check any pending crossings for this track
        # ------------------------------------------------------------------
        for line in lines:
            pkey = (camera_id, tracked.track_id, line.get("id"))
            pending = self._pending.get(pkey)

            if pending is None:
                continue

            # Update keep-alive timestamp
            pending["last_seen"] = now

            # If the track has returned to the side it started from, the
            # crossing was a graze/bounce — cancel it rather than firing.
            original_direction = pending["event"][1]["direction"]
            current_side = side_of(line, cx, cy)
            # original_direction=="in" means the track started on the negative
            # (left) side; if it's back there now, it retreated.
            retreated = (
                (original_direction == DIRECTION_IN  and current_side < 0) or
                (original_direction == DIRECTION_OUT and current_side > 0)
            )
            if retreated:
                del self._pending[pkey]
                continue

            # Distance from the stored crossing point to current anchor
            ox, oy = pending["cross_point"]
            post_travel = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5

            if post_travel >= MIN_POST_TRAVEL:
                fired.append(pending["event"])
                del self._pending[pkey]

        # ------------------------------------------------------------------
        # Phase 1: detect new intersections (skip if MIN_TRAVEL jitter)
        # ------------------------------------------------------------------
        if abs(cx - px) < MIN_TRAVEL and abs(cy - py) < MIN_TRAVEL:
            return fired

        for line in lines:
            if tracked.group == "person" and not line.get("person_enabled", True):
                continue

            if tracked.group == "vehicle" and not line.get("vehicle_enabled", False):
                continue

            if not segments_intersect(
                px, py, cx, cy,
                line["x1"], line["y1"], line["x2"], line["y2"],
            ):
                continue

            # Which way through - the side the track came FROM decides it.
            direction = DIRECTION_IN if side_of(line, px, py) < 0 else DIRECTION_OUT

            configured = line.get("direction", "both")
            if configured != "both" and configured != direction:
                continue

            event = (
                "line_crossing",
                {
                    "line_id": line.get("id"),
                    "line_name": line.get("name"),
                    "direction": direction,
                },
            )

            if MIN_POST_TRAVEL <= 0.0:
                # Instant mode — old behaviour, no waiting.
                fired.append(event)
            else:
                pkey = (camera_id, tracked.track_id, line.get("id"))
                # Only register if not already pending (don't reset the
                # cross_point if the segment re-intersects next frame).
                if pkey not in self._pending:
                    self._pending[pkey] = {
                        "event": event,
                        "cross_point": (cx, cy),
                        "last_seen": now,
                    }

        return fired

    # ----------------------------------------------------------------- prune
    def prune(self, max_age_seconds=STATE_MAX_AGE_SECONDS):
        cutoff = time.monotonic() - max_age_seconds

        stale = [
            key for key, record in self._previous.items()
            if record["last_seen"] < cutoff
        ]
        for key in stale:
            del self._previous[key]

        # Also expire pending crossings whose track has gone away.
        stale_pending = [
            key for key, record in self._pending.items()
            if record["last_seen"] < cutoff
        ]
        for key in stale_pending:
            del self._pending[key]

        return len(stale) + len(stale_pending)

    def forget_camera(self, camera_id):
        for key in [k for k in self._previous if k[0] == camera_id]:
            del self._previous[key]

        # Also clear any pending crossings for this camera so they don't fire
        # against the next camera that gets the same camera_id slot.
        for key in [k for k in self._pending if k[0] == camera_id]:
            del self._pending[key]

    def tracked_keys(self):
        return len(self._previous)
