"""
Pose-based Fall decision - the production replacement for the bbox rule.

    policy = FallPosePolicy()
    result = policy.observe(camera_id, track_id, keypoints, confidences,
                            bbox, observed_at)
    if result.event is not None:
        ...

Runs NO model. Pure geometry over COCO-17 keypoints somebody else produced,
exactly as fall_policy.py is pure geometry over boxes somebody else produced -
so it is testable without a GPU and cannot stall a camera.

WHY THIS REPLACES THE BBOX RULE
-------------------------------
fall_policy.FallEvaluator decided from bounding-box aspect ratio. On the
current evidence set that produced 16 candidates, all 16 confirmed false
positives on manual review: people bending over counters, crouching and
leaning produce the same short, wide box a fallen person does.

Offline replay over the same 24 videos found the discriminating quantity is
the TORSO ANGLE, not the box:

    known true positive        torso ~85 deg (horizontal)
    worst false positive       torso  58.8 deg at its candidate moment,
                               61.5 deg anywhere in a +/-1 s window

The thresholds below are the ones that experiment validated. They are an
INITIAL production configuration, not universal constants: they rest on 16
known negatives and ONE known positive.

TORSO ANGLE - the definition must not drift
-------------------------------------------
Identical to the offline experiment (`fall_pose_experiment.angle_from_vertical`)
and to the same helper in the Fall/Fence prototypes:

    shoulder_center = midpoint(kp[5], kp[6])     # confident points only
    hip_center      = midpoint(kp[11], kp[12])
    v               = hip_center - shoulder_center
    torso_angle     = degrees(atan2(|v.x|, |v.y|))

0 = upright, 90 = horizontal. Measured in IMAGE space, so it is camera-geometry
dependent - an overhead camera projects a horizontal body onto a near-vertical
axis. That is a known limitation, not an oversight.

Changing this definition invalidates the offline validation.
"""

import math
import os
import threading
import time
from collections import deque


def _float_env(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


#: Unchanged from the bbox implementation - same feature key, same event type.
FALL_CAMERA_FEATURE = "fall_detection"
EVENT_TYPE = "fall_detected"
PERSON_GROUP = "person"

#: Validated offline. Do not tune without re-running the replay.
TORSO_FALL_DEG = _float_env("FALL_POSE_TORSO_DEG", 75.0)

#: The FULL-BODY axis, ankle -> shoulder. A person who bends at the waist or
#: sits leaning forward puts the hip->shoulder SEGMENT horizontal while their
#: body as a whole stays upright, and the torso angle alone cannot tell that
#: apart from lying on the floor. Measured on real production footage:
#:
#:     bending for a bag   torso 84-88   trunk 26-33
#:     seated at a desk    torso 76      trunk 63
#:     known true fall     torso 78      trunk 82-90  (112/112 frames >= 75)
#:
#: Same convention and same value as the torso gate: 0 = upright, 90 = flat.
#: Deliberately the same 75.0, so there is one horizontality threshold to
#: reason about rather than two.
TRUNK_FALL_DEG = _float_env("FALL_POSE_TRUNK_DEG", 75.0)

#: ---- Transition evidence (the temporal discriminator) --------------------
#:
#: A horizontal body is not by itself a fall. A guard reclining far back in an
#: office chair is genuinely horizontal on BOTH axes - measured on 12 live
#: production events, torso 75.3-80.1 and trunk 75.4-88.4 - so no threshold on
#: those two angles can separate him from someone on the floor. What separates
#: them is history: a fall is preceded by an UPRIGHT person, a reclined chair
#: is not. Across those 12 events the lowest torso angle in the 3 s before the
#: trigger was 63.6-80.9 deg, i.e. the man was never upright; across the three
#: true falls the person was walking (torso 0.8-39.3) shortly before.
#:
#: So an event now additionally requires that this same track was recently
#: seen clearly upright. This is a REQUIREMENT, not an alternative branch -
#: it has to be, because the chair events already satisfy the geometry.
UPRIGHT_ANGLE_DEG = _float_env("FALL_POSE_UPRIGHT_DEG", 45.0)
UPRIGHT_AR = _float_env("FALL_POSE_UPRIGHT_AR", 0.9)
UPRIGHT_WINDOW_SECONDS = _float_env("FALL_POSE_UPRIGHT_WINDOW", 3.0)

#: ---- Ground state ---------------------------------------------------------
#:
#: With transition evidence required, the geometry can also recognise a
#: genuine prone posture that the torso gate alone misses. fall3.mp4 is a real
#: fall on snow where the person lies propped on their arms: torso peaks at
#: ~72 deg and trunk at ~76, both under the 75/75 gate, yet they are plainly
#: on the ground and stay there for seconds. A wider-than-tall box is what
#: makes that state safe to accept - the bending cases that the trunk gate
#: was built for have upright-shaped boxes (aspect ratio 0.77 and below).
GROUND_TORSO_DEG = _float_env("FALL_POSE_GROUND_TORSO_DEG", 65.0)
GROUND_AR = _float_env("FALL_POSE_GROUND_AR", 1.0)
CONFIRM_SECONDS = _float_env("FALL_POSE_CONFIRM_SECONDS", 0.3)
MIN_CONFIDENT_KPTS = _int_env("FALL_POSE_MIN_KPTS", 8)
KPT_CONF = _float_env("FALL_POSE_KPT_CONF", 0.30)

#: Cooldown. The bbox rule had no time-based cooldown - it emitted once per
#: fall via `event_sent_for_fall`, cleared when the person stood back up. That
#: same one-event-per-fall guarantee is kept below; this is an additional wall
#: on top of it so a person who stays down cannot re-arm repeatedly.
COOLDOWN_SECONDS = _float_env("FALL_POSE_COOLDOWN_SECONDS", 60.0)

#: A confirming track that loses its pose entirely is abandoned after this.
STALE_SECONDS = _float_env("FALL_POSE_STALE_SECONDS", 3.0)
TRACK_CACHE_CAP = _int_env("FALL_POSE_TRACK_CACHE_CAP", 4096)

# COCO-17
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# ---- states (§14) -------------------------------------------------------
NORMAL = "NORMAL"
POSE_CANDIDATE = "POSE_CANDIDATE"
POSE_CONFIRMING = "POSE_CONFIRMING"
POSE_CONFIRMED = "POSE_CONFIRMED"
POSE_UNCERTAIN = "POSE_UNCERTAIN"
COOLDOWN = "COOLDOWN"


def midpoint(kpts, confs, i, j, floor=KPT_CONF):
    """Midpoint of two keypoints, or whichever single one is confident."""
    a_ok = confs[i] >= floor
    b_ok = confs[j] >= floor
    if a_ok and b_ok:
        return ((float(kpts[i][0]) + float(kpts[j][0])) / 2.0,
                (float(kpts[i][1]) + float(kpts[j][1])) / 2.0)
    if a_ok:
        return (float(kpts[i][0]), float(kpts[i][1]))
    if b_ok:
        return (float(kpts[j][0]), float(kpts[j][1]))
    return None


def angle_from_vertical(low, high):
    """Degrees between the low->high axis and image vertical. 0=upright, 90=flat."""
    if low is None or high is None:
        return None
    dx = low[0] - high[0]
    dy = low[1] - high[1]
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def joint_angle(a, b, c):
    """Interior angle at b in degrees; 180 = straight limb."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def pose_measurements(kpts, confs, bbox):
    """Every quantity §10 asks for, plus the torso angle the decision uses."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    shoulder = midpoint(kpts, confs, L_SHOULDER, R_SHOULDER)
    hip = midpoint(kpts, confs, L_HIP, R_HIP)
    knee = midpoint(kpts, confs, L_KNEE, R_KNEE)
    ankle = midpoint(kpts, confs, L_ANKLE, R_ANKLE)

    def seg_tilt(i, j):
        if confs[i] < KPT_CONF or confs[j] < KPT_CONF:
            return None
        dx = float(kpts[i][0]) - float(kpts[j][0])
        dy = float(kpts[i][1]) - float(kpts[j][1])
        if math.hypot(dx, dy) < 1e-6:
            return None
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    knees = []
    for hip_i, knee_i, ank_i in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        if all(confs[k] >= KPT_CONF for k in (hip_i, knee_i, ank_i)):
            a = joint_angle(kpts[hip_i], kpts[knee_i], kpts[ank_i])
            if a is not None:
                knees.append(a)

    return {
        "confident_keypoints": int(sum(1 for c in confs if c >= KPT_CONF)),
        "torso_angle": angle_from_vertical(hip, shoulder),
        "trunk_angle": angle_from_vertical(ankle, shoulder),
        "shoulder_angle": seg_tilt(L_SHOULDER, R_SHOULDER),
        "hip_angle": seg_tilt(L_HIP, R_HIP),
        "knee_angle_min": (min(knees) if knees else None),
        "shoulder_center": shoulder,
        "hip_center": hip,
        "knee_center": knee,
        "ankle_center": ankle,
        "body_width": width,
        "body_height": height,
        "body_aspect_ratio": width / height,
    }


class FallPoseEvent:
    """One confirmed pose fall."""

    def __init__(self, camera_id, track_id, bbox, observed_at, measurements,
                 confirmed_seconds, pose_track_iou, model_name, imgsz,
                 frame_width=None, frame_height=None, seconds_since_upright=None):
        self.camera_id = camera_id
        self.track_id = track_id
        self.bbox = bbox
        self.observed_at = observed_at
        self.measurements = measurements
        self.confirmed_seconds = confirmed_seconds
        self.pose_track_iou = pose_track_iou
        self.model_name = model_name
        self.imgsz = imgsz
        self.frame_width = frame_width
        self.frame_height = frame_height
        #: how long before the down state began this person was last upright
        self.seconds_since_upright = seconds_since_upright

    def metadata(self):
        m = self.measurements
        return {
            "track_id": self.track_id,
            # Diagnostic fields §17. All inside the existing metadata JSON;
            # no schema change, so no dashboard consumer can break.
            "detection_method": "pose",
            "method": "pose_torso_angle",
            "pose_used": True,
            "torso_angle": (None if m.get("torso_angle") is None
                            else round(m["torso_angle"], 2)),
            "torso_threshold_deg": TORSO_FALL_DEG,
            "trunk_threshold_deg": TRUNK_FALL_DEG,
            "trunk_axis_available": m.get("trunk_angle") is not None,
            "ground_torso_threshold_deg": GROUND_TORSO_DEG,
            "ground_aspect_ratio_min": GROUND_AR,
            "upright_window_seconds": UPRIGHT_WINDOW_SECONDS,
            "seconds_since_upright": (None if self.seconds_since_upright is None
                                      else round(self.seconds_since_upright, 3)),
            "pose_duration": round(self.confirmed_seconds, 3),
            "pose_confirm_seconds": CONFIRM_SECONDS,
            "confident_keypoints": m.get("confident_keypoints"),
            "min_confident_keypoints": MIN_CONFIDENT_KPTS,
            "pose_track_iou": (None if self.pose_track_iou is None
                               else round(self.pose_track_iou, 3)),
            "pose_model": self.model_name,
            "pose_imgsz": self.imgsz,
            "knee_angle_min": (None if m.get("knee_angle_min") is None
                               else round(m["knee_angle_min"], 2)),
            "trunk_angle": (None if m.get("trunk_angle") is None
                            else round(m["trunk_angle"], 2)),
            "aspect_ratio": round(m.get("body_aspect_ratio", 0.0), 3),
        }

    def scope(self):
        """Same debounce scope the bbox implementation used."""
        return f"fall:{self.track_id}"

    def __repr__(self):
        t = self.measurements.get("torso_angle")
        return (f"<FallPoseEvent {self.camera_id}/{self.track_id} "
                f"torso={t:.1f} for {self.confirmed_seconds:.2f}s>"
                if t is not None else
                f"<FallPoseEvent {self.camera_id}/{self.track_id}>")


class _Result:
    __slots__ = ("state", "event", "reason", "measurements")

    def __init__(self, state, event=None, reason="", measurements=None):
        self.state = state
        self.event = event
        self.reason = reason
        self.measurements = measurements or {}


class _TrackState:
    __slots__ = ("state", "since", "last_seen", "cooldown_until",
                 "event_sent", "uncertain_count", "last_upright")

    def __init__(self):
        self.state = NORMAL
        self.since = None            # when the torso first crossed the threshold
        self.last_seen = 0.0
        self.cooldown_until = 0.0
        self.event_sent = False
        self.uncertain_count = 0
        #: last moment this track was seen clearly upright - the transition
        #: evidence a fall needs and a reclined chair never has.
        self.last_upright = None


class FallPosePolicy:
    """The pose Fall state machine. One instance for the whole process."""

    def __init__(self, model_name="yolo26l-pose", imgsz=960):
        self._lock = threading.Lock()
        self._tracks = {}
        self.model_name = model_name
        self.imgsz = imgsz
        self.observed = 0
        self.confirmed = 0
        self.uncertain = 0
        self.errors = 0

    # ------------------------------------------------------------- observe
    def observe(self, camera_id, track_id, kpts, confs, bbox, observed_at=None,
                pose_track_iou=None, frame_width=None, frame_height=None):
        now = time.monotonic() if observed_at is None else observed_at
        key = (camera_id, track_id)
        with self._lock:
            st = self._tracks.get(key)
            if st is None:
                st = _TrackState()
                self._tracks[key] = st
        st.last_seen = now
        self.observed += 1

        m = pose_measurements(kpts, confs, bbox)

        # §15 cooldown, and the one-event-per-fall guarantee carried over
        # from the bbox implementation.
        if now < st.cooldown_until:
            st.state = COOLDOWN
            return _Result(COOLDOWN, None,
                           f"cooldown for {st.cooldown_until - now:.1f}s more", m)

        # §13 keypoint policy. Insufficient pose is NOT evidence of no fall -
        # it is recorded as POSE_UNCERTAIN, counted, and raises no alarm.
        if m["confident_keypoints"] < MIN_CONFIDENT_KPTS or m["torso_angle"] is None:
            st.since = None
            st.state = POSE_UNCERTAIN
            st.uncertain_count += 1
            self.uncertain += 1
            return _Result(
                POSE_UNCERTAIN, None,
                f"pose too sparse to judge: {m['confident_keypoints']} confident "
                f"keypoints (need {MIN_CONFIDENT_KPTS})"
                + ("" if m["torso_angle"] is not None else ", torso axis unavailable"),
                m)

        torso = m["torso_angle"]
        trunk = m.get("trunk_angle")
        aspect = m.get("body_aspect_ratio")

        # Remember when this person was last clearly UPRIGHT. This is the
        # transition evidence; it is recorded on every observation and is what
        # a fall has and a reclined chair does not.
        if (torso <= UPRIGHT_ANGLE_DEG
                and (trunk is None or trunk <= UPRIGHT_ANGLE_DEG)
                and aspect is not None and aspect <= UPRIGHT_AR):
            st.last_upright = now

        # Two ways to be "down". Both still need the transition evidence below.
        #
        #   horizontal : the original gate, unchanged - a horizontal torso AND
        #                a horizontal full-body axis when that axis exists.
        #                This is the trunk fix and it is preserved exactly.
        #   ground     : a lower torso angle, but only in a wider-than-tall
        #                body box. That is a person lying on the floor whose
        #                torso is propped up - real, and invisible to the gate
        #                above. A bend keeps an upright-shaped box, so it
        #                cannot reach this branch.
        horizontal = (torso >= TORSO_FALL_DEG
                      and (trunk is None or trunk >= TRUNK_FALL_DEG))
        ground = ((torso >= GROUND_TORSO_DEG
                   or (trunk is not None and trunk >= TRUNK_FALL_DEG))
                  and aspect is not None and aspect >= GROUND_AR)

        if not (horizontal or ground):
            st.since = None
            st.state = POSE_CANDIDATE
            st.event_sent = False
            if torso < TORSO_FALL_DEG:
                why = f"torso {torso:.1f} deg < {TORSO_FALL_DEG} deg"
            else:
                why = (f"torso {torso:.1f} deg but full-body trunk axis "
                       f"{trunk:.1f} deg < {TRUNK_FALL_DEG} deg and body box "
                       f"is not wider than tall - bent or seated, not fallen")
            return _Result(POSE_CANDIDATE, None, why, m)

        # §12 temporal confirmation on measured timestamps, never on a frame
        # count - the cameras run at 15, 25 and 30 fps and the interval varies.
        if st.since is None:
            st.since = now
        held = now - st.since
        if held < CONFIRM_SECONDS:
            st.state = POSE_CONFIRMING
            return _Result(POSE_CONFIRMING, None,
                           f"torso {torso:.1f} deg held {held:.2f}s of "
                           f"{CONFIRM_SECONDS}s", m)

        # TRANSITION EVIDENCE. A fall is a CHANGE: upright, then down. Somebody
        # who was already horizontal when we first saw them - reclined in a
        # chair, lying on a couch - never made that transition and is not a
        # fall. Measured on the 12 live chair events, none had an upright
        # observation in the 3 s before the down state began.
        if (st.last_upright is None
                or (st.since - st.last_upright) > UPRIGHT_WINDOW_SECONDS):
            st.state = POSE_CANDIDATE
            ago = ("never" if st.last_upright is None
                   else f"{st.since - st.last_upright:.1f}s ago")
            return _Result(POSE_CANDIDATE, None,
                           f"down for {held:.2f}s but this person was last "
                           f"upright {ago} (need within "
                           f"{UPRIGHT_WINDOW_SECONDS}s) - no fall transition", m)

        if st.event_sent:
            st.state = POSE_CONFIRMED
            return _Result(POSE_CONFIRMED, None, "event already sent for this fall", m)

        st.state = POSE_CONFIRMED
        st.event_sent = True
        st.cooldown_until = now + COOLDOWN_SECONDS
        self.confirmed += 1
        return _Result(
            POSE_CONFIRMED,
            FallPoseEvent(camera_id, track_id, list(bbox), now, m, held,
                          pose_track_iou, self.model_name, self.imgsz,
                          frame_width, frame_height,
                          seconds_since_upright=(st.since - st.last_upright)),
            f"torso {torso:.1f} deg >= {TORSO_FALL_DEG} deg sustained {held:.2f}s "
            f"with {m['confident_keypoints']} confident keypoints", m)

    # -------------------------------------------------------------- upkeep
    def prune(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            for key in [k for k, v in self._tracks.items()
                        if now - v.last_seen > STALE_SECONDS
                        and now >= v.cooldown_until]:
                del self._tracks[key]
            if len(self._tracks) > TRACK_CACHE_CAP:
                for key in sorted(self._tracks,
                                  key=lambda k: self._tracks[k].last_seen)[
                        :len(self._tracks) - TRACK_CACHE_CAP]:
                    del self._tracks[key]

    def state_of(self, camera_id, track_id):
        with self._lock:
            st = self._tracks.get((camera_id, track_id))
        return st.state if st else NORMAL

    def stats(self):
        with self._lock:
            n = len(self._tracks)
        return {"fall_pose_observed": self.observed,
                "fall_pose_confirmed": self.confirmed,
                "fall_pose_uncertain": self.uncertain,
                "fall_pose_errors": self.errors,
                "fall_pose_tracked": n}
