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

ZAYED 2026-09-29 - LOWER-BODY EVIDENCE (docs/ZAYED_FALL_REMEDIATION.md)
------------------------------------------------------------------------
Event 32 (camera_01, 2026-09-28 07:15:14 Dubai) raised a critical "Person Fell"
for a person bending over a desk. Replayed offline from the NVR footage it
reproduces exactly: torso 66-67 deg, box 1.05 wide-to-tall, knees visible,
ANKLES hidden by the desk. Without ankles there is no trunk axis, and then
neither rule looked below the hips: `horizontal` fell back to the torso alone,
and `ground` never consulted the lower body at all - it trusted the box shape,
and a desk that cuts off the legs turns any bend into a wide box.

Three changes, each measured on that footage:
  * the body below the hips must be down too: the ankle->shoulder axis when the
    ankles are visible (which `horizontal` already used and `ground` now uses
    too), else the KNEE->shoulder axis. The bend's
    knee axis stayed at 22-28 deg; a person lying down reads 70-90;
  * with no knee and no ankle visible, the hips AND the shoulders must have come
    down towards the floor, in the person's own upright torso lengths from their
    last upright observation. The bend's hips moved -0.07..+0.03 of one;
  * the down state must be held CONFIRM_SECONDS = 1.0 s (was 0.3 s).
The transition requirement (clearly upright shortly before) is unchanged.
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
#:
#: ZAYED: not in a classroom. The box is the TRACK's box, and a desk that hides
#: the lower legs cuts it off at the knees, so a bend over the desk is wider than
#: tall (Event 32: 1.045, replay 1.04-1.07). The ground state now also needs the
#: body below the hips to be down - see lower_body_is_down().
GROUND_TORSO_DEG = _float_env("FALL_POSE_GROUND_TORSO_DEG", 65.0)
GROUND_AR = _float_env("FALL_POSE_GROUND_AR", 1.0)

#: ZAYED: 1.0 s, was 0.3 s. On the Event 32 footage the old rule's down state
#: ran 0.9, 1.3 and 1.2 s in the three frame-sampling phases, so time alone could
#: not have separated that bend - the lower-body evidence does. What the longer
#: hold buys is margin against transients (a mis-associated pose, a tracker
#: swap), which 0.3 s - three pose samples at the live 10 fps - does not have.
#: 1.0 s is ten consecutive samples, and it stays inside the adapter's 2 s
#: CANDIDATE_HOLD_SECONDS, so a person whose box briefly narrows while lying can
#: still complete it. Longer is not free: any single non-down sample restarts the
#: hold, and the same footage shows a one-sample dip inside a sustained posture.
CONFIRM_SECONDS = _float_env("FALL_POSE_CONFIRM_SECONDS", 1.0)

#: ZAYED: with NO knee and NO ankle visible there is no lower-body axis, and the
#: torso plus a (desk-truncated) box is not evidence of lying down. Then both
#: landmarks must have come down towards the floor, measured in the person's own
#: torso length at their last upright observation, from their position there:
#:
#:     hips       >= HIP_DROP_MIN_RATIO       standing ~0.9 m, a torso ~0.5 m; lying on
#:                                            the floor they drop ~1.6 torsos
#:     shoulders  >= SHOULDER_DROP_MIN_RATIO  ~1.4 m -> ~0.2 m lying, ~2.4 torsos
#:
#: Each rules out a different look-alike. The hips rule out a bend: Event 32's
#: moved -0.07..+0.03 torsos while its shoulders came down. The shoulders rule out
#: everything that lowers the hips but keeps the upper body up: sitting down and
#: slumping onto the desk (shoulders ~0.8 m, ~1.2 torsos), kneeling (~1.6), a deep
#: squat leaning forward 70 deg (~0.5 m, ~1.8) - which is why this is 2.0, not
#: lower. Measured in IMAGE space like every other quantity here, so a fall
#: straight away from the camera shows less drop than one across its view.
HIP_DROP_MIN_RATIO = _float_env("FALL_POSE_HIP_DROP_RATIO", 1.0)
SHOULDER_DROP_MIN_RATIO = _float_env("FALL_POSE_SHOULDER_DROP_RATIO", 2.0)
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
        # ZAYED: the knee -> shoulder axis - the lower-body evidence when a desk
        # hides the ankles (see lower_body_is_down).
        "knee_axis_angle": angle_from_vertical(knee, shoulder),
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


def lower_body_evidence(m, upright):
    """ZAYED. What the body below the hips says, and how far the hips and the
    shoulders have come down since this person was last upright.

    m        pose_measurements() of this observation
    upright  (hip_y, shoulder_y, torso_length) at the last upright observation,
             or None when there has been none
    """
    if m.get("trunk_angle") is not None:
        source, angle = "ankle", m["trunk_angle"]
    elif m.get("knee_axis_angle") is not None:
        source, angle = "knee", m["knee_axis_angle"]
    else:
        source, angle = None, None
    hip_drop = shoulder_drop = None
    hip, shoulder = m.get("hip_center"), m.get("shoulder_center")
    if upright is not None and hip is not None and shoulder is not None:
        up_hip_y, up_shoulder_y, torso_length = upright
        if torso_length > 1e-6:
            # Image y grows downwards, so a positive drop is towards the floor.
            hip_drop = (hip[1] - up_hip_y) / torso_length
            shoulder_drop = (shoulder[1] - up_shoulder_y) / torso_length
    return {"lower_body_source": source, "lower_body_angle": angle,
            "hip_drop_ratio": hip_drop, "shoulder_drop_ratio": shoulder_drop}


def lower_body_is_down(evidence, min_axis_deg):
    """ZAYED. Is the body below the hips down as well as the torso?

    With an ankle or knee axis, that axis must be at least `min_axis_deg` from
    vertical - the same horizontality the rule asks of the torso. With neither,
    the hips and the shoulders must both have come down (HIP_DROP_MIN_RATIO,
    SHOULDER_DROP_MIN_RATIO). No upright reference means no evidence: no.
    """
    if evidence["lower_body_source"] is not None:
        return evidence["lower_body_angle"] >= min_axis_deg
    hip_drop, shoulder_drop = evidence["hip_drop_ratio"], evidence["shoulder_drop_ratio"]
    return (hip_drop is not None and shoulder_drop is not None
            and hip_drop >= HIP_DROP_MIN_RATIO and shoulder_drop >= SHOULDER_DROP_MIN_RATIO)


def _not_down_reason(torso, trunk, aspect, evidence):
    """ZAYED. Why this observation is not a down state, in an operator's words."""
    if torso < GROUND_TORSO_DEG and not (trunk is not None and trunk >= TRUNK_FALL_DEG):
        return f"torso {torso:.1f} deg < {GROUND_TORSO_DEG} deg"
    source = evidence["lower_body_source"]
    if source is not None:
        angle = evidence["lower_body_angle"]
        if angle < GROUND_TORSO_DEG:
            return (f"torso {torso:.1f} deg, but the {source}->shoulder axis is {angle:.1f} deg: "
                    f"the body below the hips is upright - bent or seated, not lying")
        return (f"torso {torso:.1f} deg, {source}->shoulder axis {angle:.1f} deg, but the box is "
                f"not wider than tall ({aspect:.2f} < {GROUND_AR}) - bent or seated, not fallen")
    hip_drop, shoulder_drop = evidence["hip_drop_ratio"], evidence["shoulder_drop_ratio"]
    if hip_drop is None:
        return (f"torso {torso:.1f} deg with no knee or ankle visible and no upright reference - "
                f"no evidence of lying down")
    return (f"torso {torso:.1f} deg with no knee or ankle visible; hips dropped {hip_drop:+.2f} and "
            f"shoulders {shoulder_drop:+.2f} upright torso lengths (need {HIP_DROP_MIN_RATIO} and "
            f"{SHOULDER_DROP_MIN_RATIO}) - leaning over, not on the floor")


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
            # ZAYED: the lower-body evidence the decision rested on.
            "down_rule": m.get("down_rule"),
            "lower_body_evidence": m.get("lower_body_source") or "none",
            "lower_body_axis_angle": (None if m.get("lower_body_angle") is None
                                      else round(m["lower_body_angle"], 2)),
            "knee_axis_angle": (None if m.get("knee_axis_angle") is None
                                else round(m["knee_axis_angle"], 2)),
            "hip_drop_ratio": (None if m.get("hip_drop_ratio") is None
                               else round(m["hip_drop_ratio"], 3)),
            "shoulder_drop_ratio": (None if m.get("shoulder_drop_ratio") is None
                                    else round(m["shoulder_drop_ratio"], 3)),
            "hip_drop_threshold": HIP_DROP_MIN_RATIO,
            "shoulder_drop_threshold": SHOULDER_DROP_MIN_RATIO,
            "policy_revision": "zayed-lower-body-2026-09-29",
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
                 "event_sent", "uncertain_count", "last_upright", "upright_ref")

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
        #: ZAYED: (hip_y, shoulder_y, torso_length) at that moment - the person's
        #: own upright geometry, which lower_body_evidence() measures drops from.
        self.upright_ref = None


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
            # ZAYED: and WHERE this person's hips and shoulders were, so a later
            # observation can say how far they have come down since.
            hip, shoulder = m["hip_center"], m["shoulder_center"]
            st.upright_ref = (hip[1], shoulder[1],
                              math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1]))

        # Two ways to be "down". Both still need the transition evidence below.
        #
        #   horizontal : a horizontal torso AND a horizontal body below the hips.
        #   ground     : a lower torso angle, but only in a wider-than-tall box -
        #                a person lying on the floor whose torso is propped up.
        #
        # ZAYED: both now also need lower_body_is_down(). Dubai let `horizontal`
        # fall back to the torso alone when the ankles were hidden, and `ground`
        # never looked below the hips, trusting the box shape - which a desk
        # hiding the legs defeats (Event 32). The ankle axis is still the first
        # choice: with the legs visible `horizontal` is unchanged, while `ground`
        # now also needs that axis at 65 deg, so a reach to the floor with the
        # arms out (a wide box over upright legs) no longer passes either.
        evidence = lower_body_evidence(m, st.upright_ref)
        m.update(evidence)
        horizontal = (torso >= TORSO_FALL_DEG
                      and lower_body_is_down(evidence, TRUNK_FALL_DEG))
        ground = ((torso >= GROUND_TORSO_DEG
                   or (trunk is not None and trunk >= TRUNK_FALL_DEG))
                  and aspect is not None and aspect >= GROUND_AR
                  and lower_body_is_down(evidence, GROUND_TORSO_DEG))

        if not (horizontal or ground):
            st.since = None
            st.state = POSE_CANDIDATE
            st.event_sent = False
            return _Result(POSE_CANDIDATE, None,
                           _not_down_reason(torso, trunk, aspect, evidence), m)
        m["down_rule"] = "horizontal" if horizontal else "ground"

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
            f"{m['down_rule']}: torso {torso:.1f} deg sustained {held:.2f}s "
            f"with {m['confident_keypoints']} confident keypoints, lower body "
            f"{m.get('lower_body_source') or 'hidden'}", m)

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
