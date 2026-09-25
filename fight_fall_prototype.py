"""
Pose-Based Fight (Violence) + Fall Detection — yolo26m-pose TensorRT Engine
============================================================================

Multi-source video runner for Fall_detection/fightfalldetection.py's analytic
logic, built the same way test_fall_det.py tests falldetection.py's: the
geometry/motion functions (analyze_posture, limb_velocities, _strikes,
detect_fights, and every FALL_*/STRIKE_*/FIGHT_* threshold) are copied
VERBATIM from fightfalldetection.py - not reimplemented, not approximated -
only the detector/tracker underneath is swapped for this project's own
stack, matching test_fall_det.py's own precedent of doing exactly that for
falldetection.py. Fight detection specifically is ported from
AI_inferencing/fightfalldetection.py's re-tuned, strike-based version (not
the original Fall_detection/fightfalldetection.py copy this script started
from) - see STRIKE_SPEED_PS's comment below for why.

Accepts any mix of:
  - MediaMTX camera names  (cam-r01, cam-r09, ...)
  - RTSP URLs              (rtsp://host/path)
  - Local video files      (Fall_detection/data/fight.mp4, ...)

Sources are read via cctv.CameraStream (NVDEC RTSP) for RTSP/camera names, or
cv2.VideoCapture (looping) for local files - same dual-source parse_sources()
test_fall_det.py already uses.

Both this script and test_fall_det.py now use tracking.TrackManager (the same
production tracker the rest of this project's test scripts already use) for
real per-person identity - test_fall_det.py's own port went in for the same
reason this script always needed it: model.predict() alone never sets
result.boxes.id, so without a real tracker every person in frame pools into
one shared anonymous history and any per-person or per-pair reasoning is
impossible. For fight detection specifically this is not optional - a
genuinely fast/serious limitation, not a nice-to-have: detect_fights() only
pairs people that have a real, distinct track_id, so an all-None people list
means the candidate-pair list is always empty and FIGHT can never fire.
TrackManager is fed straight from the pose model's own person boxes (one
model call per frame, no separate detection engine), then matched back to
that frame's keypoints by IOU, since TrackManager doesn't carry a keypoints
payload through itself.

============================================================
UPDATED FOR fightfalldetection.py's LATEST (re-tuned) VERSION
============================================================
analyze_posture/limb_velocities/the core geometry were already in sync with
that file. Two mechanisms were newly ported in this pass, both aimed at the
same failure mode from different angles - a crowded or overhead scene, where
single-frame geometry alone is not enough evidence:

  FALL: was a flat windowed hold (fall alarmed once >=FALL_MIN_DOWN of the
  last FALL_WINDOW frames read as down). Now a person is only flagged once
  they make a real upright -> ground TRANSITION: a per-track queue of
  'standing'/'lying'/'other' posture states, SUBSAMPLED to FALL_STATE_HZ (not
  every frame, so a small queue still spans several real seconds) is checked
  for a recent 'standing' sample before a run of 'lying' samples latches the
  alarm. Someone already sitting or lying on the floor when first seen -
  indistinguishable from a genuine fall by single-frame geometry on an
  overhead camera - is never flagged, because they were never seen standing.
  Same idea as test_fall_det.py's own FallMonitor (an upright person going
  down fast is the real signal), a different implementation: FallMonitor
  reasons about raw bbox aspect ratio directly over a time-windowed span,
  this reasons over posture-CLASSIFIED, subsampled states - ported from
  fightfalldetection.py as-is rather than reconciled into one shared
  mechanism, since the two scripts' source analytics remain deliberately
  separate ports (see the top of this docstring).

  FIGHT: was one-sided (either person landing a single detected strike within
  the window was enough). Now MUTUAL by default (FIGHT_MUTUAL): both people
  must land FIGHT_MIN_STRIKES_EACH strikes on EACH OTHER within the window, a
  much more specific signal of an actual exchange than one side's fast-limb
  reading alone - which in a crowd is exactly what key-point bleed between
  overlapping neighbours produces. Pairs where either skeleton has fewer than
  FIGHT_MIN_KPTS confident key-points are skipped before the strike test even
  runs, for the same crowd-robustness reason.

Usage
-----
    # Default: the known fight/fall sample clips (no camera needed to verify
    # both alarms actually fire)
    aienv/bin/python fight_fall_prototype.py

    # Live cameras
    CAMERAS="cam-r01,cam-r09" aienv/bin/python fight_fall_prototype.py
    CAMERAS="cam-r01,Fall_detection/data/fall.mp4" aienv/bin/python fight_fall_prototype.py

    # Tune inference FPS and pose imgsz
    TARGET_FPS=5 POSE_IMGSZ=640 CAMERAS=cam-r09 aienv/bin/python fight_fall_prototype.py

q / Esc to quit. Nothing is sent; this script never talks to the dashboard
and does not touch multicam_inf.py, test_fall_det.py, falldetection.py, or
Fall_detection/fightfalldetection.py.

A fully-annotated frame (skeletons, boxes, fight links, alert banner - the
same frame shown on screen, at full source resolution) is saved to
Fall_detection/output/fight_frames/ on a confirmed FIGHT or a confirmed FALL
(same folder for both). One frame per newly-triggered pair/track per episode,
not one per frame of an ongoing event: a pair/track already saved for the
current episode is skipped until it drops out of confirmed_fight_ids/the
confirmed-fall set, so a later separate event involving the same person(s)
is saved again instead of being silently ignored forever.
"""

import os
import sys
import time
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent   # AI_inferencing/
sys.path.insert(0, str(BASE_DIR))

import tracking                              # noqa: E402
from cctv import CameraStream                # noqa: E402


# ============================================================
# PATHS
# ============================================================

MODELS_DIR = BASE_DIR / "models"

# Pose engine - shared with cctv_monitor.py/test_fall_det.py/test_behavior.py
POSE_WEIGHTS_PT        = str(MODELS_DIR / "yolo26m-pose.pt")
POSE_ENGINE_PATH       = str(MODELS_DIR / "yolo26m-pose.engine")
POSE_ENGINE_MAX_BATCH  = 19
POSE_ENGINE_BATCH_FILE = POSE_ENGINE_PATH + ".batch"
POSE_IMGSZ = int(os.getenv("POSE_IMGSZ", "640"))

FALL_DETECTION_DIR = BASE_DIR / "Fall_detection"

# Fully-annotated frames get saved here on a confirmed FIGHT or FALL - same
# folder for both, one per newly-triggered pair/track per episode, not one
# per frame of an ongoing event (see the save logic in main() for why).
# Fall_detection/output/ already exists as this project's convention for
# this analytic's saved output; the "fight_frames" name predates fall saving
# also landing here - left as-is rather than renaming the directory.
ALERT_OUTPUT_DIR = FALL_DETECTION_DIR / "output" / "fight_frames"


# ============================================================
# CAMERA / SOURCE CONFIGURATION
# ============================================================

MEDIAMTX_BASE = os.getenv("MEDIAMTX_BASE", "rtsp://10.232.7.151:18554")
# Default to the known sample clips fightfalldetection.py's own thresholds
# were tuned against (see STRIKE_SPEED_PS's comment below) - lets both
# alarms be visually confirmed correct with no camera/live behaviour needed.
# Point CAMERAS at real cameras once that's confirmed, e.g. CAMERAS=cam-r09.
#,cam-r18,cam-r19,cam-r03
CAMERAS = os.getenv("CAMERAS", "cam-r15")

# Target inference FPS per source
TARGET_FPS = float(os.getenv("TARGET_FPS", "10.0"))
FRAME_INTERVAL = 1.0 / TARGET_FPS

DISPLAY_WIDTH  = int(os.getenv("DISPLAY_WIDTH",  "960"))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", "540"))

# Tracking - the same values finalized in test_track.py and already used by
# the rest of this project's test scripts.
TRACK_MAX_DISAPPEARED = int(os.getenv("TRACK_MAX_DISAPPEARED", "300"))
TRACK_DIST_THRESH = float(os.getenv("TRACK_DIST_THRESH", "150"))
TRACK_IOU_THRESH = float(os.getenv("TRACK_IOU_THRESH", "0.30"))
TRACK_DUPLICATE_IOU_THRESH = float(os.getenv("TRACK_DUPLICATE_IOU_THRESH", "0.30"))

# Minimum IOU between a TrackManager box and a pose-model box this frame to
# accept that match and use its keypoints - both come from the SAME model
# call's own detections, so a true match is normally near-perfect; this only
# guards against a track coasting on a stale box (no fresh detection this
# frame) being paired with an unrelated person's keypoints.
POSE_MATCH_IOU = float(os.getenv("POSE_MATCH_IOU", "0.3"))


# ============================================================
# FALL + FIGHT THRESHOLDS
# (verbatim from Fall_detection/fightfalldetection.py - not retuned here)
# ============================================================

PERSON_CONF = 0.35       # min person detection confidence
KPT_CONF    = 0.30       # min keypoint confidence to trust

IMG_SIZE = POSE_IMGSZ    # inference resolution matches engine export size

# --- Fall geometry thresholds (single frame) --------------------------------
TORSO_HORIZONTAL_DEG = 60.0     # (60, not 55: a real fall's torso reaches 70-90;
                                # 55 was borderline and briefly flagged bending
                                # /standing people with noisy key-points)
TORSO_UPRIGHT_DEG    = 35.0
FALL_BBOX_RATIO   = 1.10
FALL_SPREAD_RATIO = 1.10
KNEE_BENT_DEG = 130.0
# A person lying on the ground has a box clearly WIDER than tall - a direct
# "on the ground" signal on its own, so the fall alarm holds even if the
# torso angle is noisy or the person props themselves up slightly.
FALL_STRONG_WIDE = 1.35

# --- Fight / violence thresholds (motion + interaction) ---------------------
FIGHT_PROXIMITY_RATIO = 0.70
FIGHT_BOX_IOU = 0.02
# A fight is not merely "two people close + fast motion" (that also describes
# people walking past each other, especially in low-frame-rate CCTV where
# every limb looks fast). A real strike is a fast limb that actually LANDS ON
# the other person. So the raw violent test for a close pair is: a wrist/ankle
# of one person is moving faster than STRIKE_SPEED_PS AND that same limb is
# within STRIKE_REACH of the OTHER person's torso.
#
# Speed is measured in body-heights PER SECOND (multiplied by fps, TARGET_FPS
# here - see limb_velocities()), so the same threshold works whether the
# camera runs at 10 fps CCTV or 30 fps video. Ported from
# AI_inferencing/fightfalldetection.py's own re-tuning, which specifically
# validated this against a busy 10 fps CCTV clip of people walking near each
# other (data/cam1.avi) to confirm it stays quiet there while still firing on
# real fighting (data/fight.mp4) - the exact frame rate and scene type this
# script itself runs at.
STRIKE_SPEED_PS = 1.5           # limb speed (body-heights / second) => "fast"
STRIKE_REACH = 0.55             # fast limb within this * body-size of other torso

# ...and the limb has to be ARRIVING at that torso rather than leaving it.
#
# The two gates above say "fast" and "near". Neither says which way the limb is
# going, because limb_velocities used to reduce the displacement to a scalar
# with np.linalg.norm before any decision could see it. The comment above
# already stated the intent - a real strike "actually LANDS ON the other
# person" - but nothing in the code tested for landing.
#
# Live consequence, CAM-R09 2026-09-10 14:43:11 (EVT-F6F54CAF, CRITICAL): a
# woman walked across the lobby holding a phone to her ear. Replaying that
# event's own evidence through this file reproduced four "strikes", and every
# one of them was travelling AWAY from the other person:
#
#   t=13.07  L_wrist  speed 2.904  reach 0.451   approach -0.96
#   t=13.47  L_wrist  speed 1.838  reach 0.399   approach -0.90
#   t=13.60  L_wrist  speed 1.520  reach 0.276   approach -0.11  (toward own head +0.94)
#   t=14.00  R_ankle  speed 1.640  reach 0.424   approach -0.26
#
# The first of those started 69.5 px from the other person's torso and ended
# 251.3 px away. It was leaving, and it was counted as a blow landing.
#
# approach = |previous limb - torso| - |current limb - torso|, divided by the
# distance the limb actually travelled. +1 is straight at the torso, -1 is
# straight away, and 0.0 - the value used - is the neutral boundary: the limb
# must end nearer the torso than it began, by any margin at all.
#
# This is NOT a sensitivity knob. Speed and reach still decide how hard a
# motion has to be; this only removes motions that were receding, which cannot
# be blows landing. Measured on the corpus in violence_audit/: genuine fights
# stay 4/4 detected, the production false positive goes to zero.
STRIKE_APPROACH_FRAC = 0.0

# --- Temporal confirmation --------------------------------------------------
FALL_WINDOW = 12
FALL_MIN_DOWN = 5
FALL_HOLD_FRAMES = 12
# Fight window is time-based (seconds), converted to frames via TARGET_FPS
# below - so it behaves the same real-world duration on any source, unlike a
# flat frame count.
FIGHT_WINDOW_SEC = 1.2
FIGHT_MIN_VIOLENT = 3           # strike frames needed within that window (one-sided mode)
FIGHT_COOLDOWN_FRAMES = 15
HISTORY_LEN = 12

# --- Robustness for overhead / crowded cameras (ported from fightfalldetection.py's
# re-tuned version - NOT in the copy this script started from) --------------
# On a high-mounted overhead camera in a room where people normally sit and lie
# on the floor, single-frame geometry cannot tell a seated person from a fallen
# one (both look "wide" from above). The fix that needs no training: treat a
# fall as a TRANSITION - a person who was recently UPRIGHT (standing) and then
# dropped to the ground. Someone already sitting/lying was never upright, so
# they are not flagged. The alarm then holds while they remain down.
# Set to False for clean side-view cameras where standing is the norm and you
# want to catch a person who is already on the ground when they first appear.
FALL_REQUIRE_TRANSITION = True
# The per-person posture history is SUBSAMPLED to a fixed rate (not every
# frame) so a small queue spans several SECONDS of real time. We record
# FALL_STATE_HZ states per second and keep FALL_STATE_SAMPLES of them.
# Default: 5 Hz x 20 samples = ~4 seconds of state history. Falls play out
# over ~1 s, so 5 Hz is plenty for tracking posture, and the long window
# gives a dependable "was this person standing recently?" memory.
# NOTE: fight strike detection still runs on EVERY frame - it needs the fast
# frame-to-frame motion to measure punches/kicks, so it is not subsampled.
FALL_STATE_HZ = 5              # posture states sampled per second
FALL_STATE_SAMPLES = 20        # states kept in the sliding queue (20 = ~4 s @ 5 Hz)
FALL_RECENT_SAMPLES = 4        # inspect the last N samples to decide "down now"
FALL_MIN_LYING = 3             # >= this many of those must be 'lying'
FALL_CLEAR_SAMPLES = 3         # consecutive non-lying samples that clear the fall

# In a crowd, bounding boxes overlap and key-points bleed between neighbouring
# bodies, so a one-sided "strike" is unreliable. Require a genuine two-way
# exchange - BOTH people land strikes on each other within the window - and
# ignore people whose skeleton is too incomplete to trust. Set FIGHT_MUTUAL to
# False for the simpler one-sided rule (fine on clean two-person clips).
FIGHT_MUTUAL = True
FIGHT_MIN_STRIKES_EACH = 2      # strikes each side must land within the window
FIGHT_MIN_KPTS = 8              # min confident key-points per person to judge a fight

# All sources share this script's own TARGET_FPS throttle (not each source's
# native fps - we're a throttled consumer, not processing every native
# frame), so ONE frame-count window works for every source, unlike the
# original single-stream script which derived it from the open video's own
# reported fps. Same reasoning applies to the fall-state subsample stride
# below - both are now derived from TARGET_FPS instead of a per-video fps.
FIGHT_WINDOW_FRAMES = max(FIGHT_MIN_VIOLENT, round(FIGHT_WINDOW_SEC * TARGET_FPS))
FALL_STATE_STRIDE = max(1, round(TARGET_FPS / FALL_STATE_HZ))

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]

LIMB_KPTS = (9, 10, 15, 16)     # left/right wrist, left/right ankle

COLOR_OK = (0, 200, 0)
COLOR_WATCH = (0, 200, 255)
COLOR_FALL = (0, 0, 255)
COLOR_FIGHT = (0, 0, 255)
COLOR_LINK = (0, 0, 255)

SHOW_POSTURE_LABELS = False


# ============================================================
# GEOMETRY HELPERS  (verbatim from fightfalldetection.py)
# ============================================================

def _valid(pt_conf):
    return pt_conf is not None and pt_conf >= KPT_CONF


def _midpoint(kpts, confs, i, j):
    a_ok, b_ok = _valid(confs[i]), _valid(confs[j])
    if a_ok and b_ok:
        return (kpts[i] + kpts[j]) / 2.0
    if a_ok:
        return kpts[i].copy()
    if b_ok:
        return kpts[j].copy()
    return None


def _angle_from_vertical(p_low, p_high):
    if p_low is None or p_high is None:
        return None
    dx = float(p_high[0] - p_low[0])
    dy = float(p_high[1] - p_low[1])
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def _joint_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _box_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def _box_size(bbox):
    x1, y1, x2, y2 = bbox
    return float(np.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1)))


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-6))


# ============================================================
# POSTURE ANALYSIS  (verbatim from fightfalldetection.py, current/improved
# version - includes FALL_STRONG_WIDE, not test_fall_det.py's older copy)
# ============================================================

def analyze_posture(kpts, confs, bbox):
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    bbox_ratio = box_w / box_h

    shoulder = _midpoint(kpts, confs, 5, 6)
    hip = _midpoint(kpts, confs, 11, 12)
    ankle = _midpoint(kpts, confs, 15, 16)

    torso_angle = _angle_from_vertical(hip, shoulder)
    leg_angle = _angle_from_vertical(ankle, hip)
    knee_l = _joint_angle(kpts[11], kpts[13], kpts[15]) if all(
        _valid(confs[k]) for k in (11, 13, 15)) else None
    knee_r = _joint_angle(kpts[12], kpts[14], kpts[16]) if all(
        _valid(confs[k]) for k in (12, 14, 16)) else None
    knee_angles = [k for k in (knee_l, knee_r) if k is not None]
    knee_angle = min(knee_angles) if knee_angles else None

    valid_pts = np.array([kpts[i] for i in range(len(kpts)) if _valid(confs[i])])
    if len(valid_pts) >= 2:
        x_span = float(valid_pts[:, 0].max() - valid_pts[:, 0].min())
        y_span = float(valid_pts[:, 1].max() - valid_pts[:, 1].min())
        spread_ratio = x_span / (y_span + 1e-6)
    else:
        spread_ratio = bbox_ratio

    metrics = {
        "torso_angle": torso_angle, "leg_angle": leg_angle,
        "knee_angle": knee_angle, "bbox_ratio": bbox_ratio,
        "spread_ratio": spread_ratio,
    }

    torso_horizontal = (torso_angle is not None) and (torso_angle >= TORSO_HORIZONTAL_DEG)
    torso_upright = (torso_angle is not None) and (torso_angle <= TORSO_UPRIGHT_DEG)
    body_wide = (bbox_ratio >= FALL_BBOX_RATIO) or (spread_ratio >= FALL_SPREAD_RATIO)
    # BOUNDING BOX only, not key-point spread: spread_ratio is easily inflated
    # by a single stray/jumbled key-point, which briefly made standing people
    # look "lying".
    strong_wide = bbox_ratio >= FALL_STRONG_WIDE
    knees_folded = (knee_angle is not None) and (knee_angle < KNEE_BENT_DEG)

    stacked = False
    if hip is not None and ankle is not None:
        vertical_drop = float(ankle[1] - hip[1])
        stacked = vertical_drop > 0.20 * box_h

    prayer_or_kneel = knees_folded and stacked and not strong_wide and not body_wide

    if prayer_or_kneel:
        label, is_fall = "Kneeling/Praying", False
    elif strong_wide or (torso_horizontal and body_wide):
        label, is_fall = "FALL", True
    elif torso_upright and not body_wide and not knees_folded:
        label, is_fall = "Standing", False
    else:
        label, is_fall = "Sitting/Bending", False

    return {"label": label, "is_fall": is_fall, "metrics": metrics}


def posture_state(info):
    """Collapse a frame's posture into one of three tracked states.

    These are the symbols pushed into each person's sliding state-queue:
        'standing' -> clearly upright
        'lying'    -> on the ground / sprawled (raw single-frame fall geometry)
        'other'    -> sitting / bending / kneeling / praying (neither of above)

    The fall decision then looks for a 'standing' -> 'lying' TRANSITION within
    the queue, so a person who is merely sitting/lying the whole time (never
    'standing') is never treated as a fall.
    """
    if info["label"] == "Standing":
        return "standing"
    if info["is_fall"]:
        return "lying"
    return "other"


# ============================================================
# LIMB MOTION  (ported from AI_inferencing/fightfalldetection.py's re-tuned
# strike-based version - see STRIKE_SPEED_PS/STRIKE_REACH's comment above for
# why this replaced the older flat-speed-threshold approach)
# ============================================================

class LimbMotion(float):
    """A limb speed that remembers the displacement it was measured from.

    A float SUBCLASS, deliberately: every caller that carries a limb-velocity
    value around keeps working untouched - it still is the speed, and compares,
    prints and serialises like one - while _strikes, the only function that
    needs the direction, can read .dx/.dy off it.
    """
    __slots__ = ("dx", "dy")

    def __new__(cls, speed, dx, dy):
        self = float.__new__(cls, speed)
        self.dx = float(dx)
        self.dy = float(dy)
        return self


def limb_velocities(prev_kpts, prev_confs, kpts, confs, bbox, fps):
    """Per-limb speed for wrists/ankles, in body-heights PER SECOND.

    Returns a dict {limb_index: LimbMotion}. The value is the key-point
    displacement since the previous frame, normalized by body size
    (scale-invariant) and multiplied by fps (frame-rate-invariant).
    Punches/kicks produce large values; standing or slow walking produce small
    ones. The displacement that produced the speed is carried alongside it so
    _strikes can tell an arriving limb from a departing one - see
    STRIKE_APPROACH_FRAC.
    """
    if prev_kpts is None:
        return {}
    scale = _box_size(bbox)
    vel = {}
    for k in LIMB_KPTS:
        if _valid(confs[k]) and _valid(prev_confs[k]):
            cur = np.asarray(kpts[k], dtype=float)
            prv = np.asarray(prev_kpts[k], dtype=float)
            d = float(np.linalg.norm(cur - prv))
            vel[k] = LimbMotion((d / (scale + 1e-6)) * fps,
                                cur[0] - prv[0], cur[1] - prv[1])
    return vel


def _approach_fraction(motion, limb_xy, target):
    """How much of this limb's travel went into closing on ``target``.

    +1.0 straight at the torso, 0.0 neither nearer nor further, -1.0 straight
    away. None when there is no displacement to judge, in which case the caller
    leaves the decision exactly as it was before this gate existed.
    """
    dx = getattr(motion, "dx", None)
    dy = getattr(motion, "dy", None)
    if dx is None or dy is None:
        return None
    disp = np.array([dx, dy], dtype=float)
    travelled = float(np.linalg.norm(disp))
    if travelled < 1e-6:
        return None
    now = np.asarray(limb_xy, dtype=float)
    before = now - disp
    closed = (float(np.linalg.norm(before - target))
              - float(np.linalg.norm(now - target)))
    return closed / travelled


def _torso_center(kpts, confs):
    """Mid-point between the shoulder centre and hip centre (body core)."""
    sh = _midpoint(kpts, confs, 5, 6)
    hp = _midpoint(kpts, confs, 11, 12)
    if sh is not None and hp is not None:
        return (sh + hp) / 2.0
    return sh if sh is not None else hp


def _visible_kpts(confs):
    """Number of key-points whose confidence clears the threshold."""
    return int(sum(1 for j in range(len(confs)) if _valid(confs[j])))


def _strikes(kpts, vel, target_torso, body_size):
    """True if any fast limb of this person LANDS ON ``target_torso``.

    Three questions, not two: fast enough, near enough, and travelling into
    that body rather than out of it. The third is what separates a punch from
    a hand withdrawing, a hand rising to its owner's own ear, and a footfall
    carrying somebody past - all of which are fast and all of which happen
    within half a body diagonal of anyone standing nearby.
    """
    if target_torso is None or not vel:
        return False
    target = np.asarray(target_torso, dtype=float)
    for j, speed in vel.items():
        if speed < STRIKE_SPEED_PS:
            continue
        limb = np.asarray(kpts[j], dtype=float)
        reach = float(np.linalg.norm(limb - target)) / (body_size + 1e-6)
        if reach > STRIKE_REACH:
            continue
        approach = _approach_fraction(speed, limb, target)
        # No measurable displacement: decide as this function always has.
        if approach is not None and approach < STRIKE_APPROACH_FRAC:
            continue
        return True
    return False


def detect_fights(people, limb_vel):
    """Evaluate every physically close pair of people for an actual strike.

    close_pairs : list of (id_a, id_b, center_a, center_b, a_hits_b, b_hits_a)
        one entry per physically close pair; ``a_hits_b`` is True when A's fast
        limb lands on B this frame, and ``b_hits_a`` the reverse. Windowed
        temporal confirmation (and the one-sided vs mutual decision) is done by
        the caller. Pairs where either skeleton is too incomplete to trust are
        skipped (common source of crowd false-positives).
    """
    close_pairs = []
    identifiable = [p for p in people if p[0] is not None]
    for (ta, ka, ca, ba), (tb, kb, cb, bb) in combinations(identifiable, 2):
        # Skip pairs with unreliable skeletons (occluded / distant / crowded).
        if _visible_kpts(ca) < FIGHT_MIN_KPTS or _visible_kpts(cb) < FIGHT_MIN_KPTS:
            continue

        ca_c, cb_c = _box_center(ba), _box_center(bb)
        mean_size = 0.5 * (_box_size(ba) + _box_size(bb))
        dist = float(np.linalg.norm(ca_c - cb_c))
        proximity = dist / (mean_size + 1e-6)
        iou = _box_iou(ba, bb)

        close = (proximity <= FIGHT_PROXIMITY_RATIO) or (iou >= FIGHT_BOX_IOU)
        if not close:
            continue

        tgt_a = _torso_center(ka, ca)
        tgt_b = _torso_center(kb, cb)
        a_hits_b = _strikes(ka, limb_vel.get(ta, {}), tgt_b, mean_size)
        b_hits_a = _strikes(kb, limb_vel.get(tb, {}), tgt_a, mean_size)
        close_pairs.append((ta, tb, ca_c, cb_c, a_hits_b, b_hits_a))

    return close_pairs


# ============================================================
# DRAWING  (verbatim from fightfalldetection.py)
# ============================================================

def draw_person(frame, kpts, confs, bbox, label, color, track_id=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    for a, b in SKELETON:
        if _valid(confs[a]) and _valid(confs[b]):
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pa, pb, color, 2)
    for i in range(len(kpts)):
        if _valid(confs[i]):
            cv2.circle(frame, (int(kpts[i][0]), int(kpts[i][1])), 3, color, -1)
    tag = f"{label}" if track_id is None else f"{track_id} {label}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 3, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_fight_links(frame, fight_pairs):
    for _ta, _tb, ca, cb in fight_pairs:
        pa = (int(ca[0]), int(ca[1]))
        pb = (int(cb[0]), int(cb[1]))
        cv2.line(frame, pa, pb, COLOR_LINK, 3)


def draw_alert_banner(frame, texts):
    if not texts:
        return
    h, w = frame.shape[:2]
    band = 45
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, band * len(texts)), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (12, 32 + band * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def decide_display(info, confirmed_fall, in_fight):
    if in_fight:
        return "FIGHT", COLOR_FIGHT
    if confirmed_fall:
        return "FALL", COLOR_FALL
    if SHOW_POSTURE_LABELS:
        if info["label"] == "FALL":
            return "Falling?", COLOR_WATCH
        if info["label"] in ("Kneeling/Praying", "Sitting/Bending"):
            return info["label"], COLOR_WATCH
        return info["label"], COLOR_OK
    if info["label"] == "FALL":
        return "Falling?", COLOR_WATCH
    return "OK", COLOR_OK


# ============================================================
# TRACK MATCHING
#
# The pose model's own person detections (this frame's boxes) go into OUR
# tracker to get stable, genuinely-distinct track_ids; keypoints don't travel
# through TrackManager (it only knows boxes/class/confidence), so each
# tracked box is matched back to this frame's pose detections by IOU - both
# came from the SAME model call, so a true match is normally near 1.0.
# Returns the exact (track_id, kpts, confs, bbox) shape
# fightfalldetection.py's own extract_people() produces, so every function
# above runs completely unchanged on it.
# ============================================================

def build_people(track_manager, source_id, result):
    if result.boxes is None or len(result.boxes) == 0:
        return []

    raw_boxes = result.boxes.xyxy.cpu().numpy()
    raw_confs = result.boxes.conf.cpu().numpy()

    if result.keypoints is not None:
        kpts_xy = result.keypoints.xy.cpu().numpy()
        kpts_cf = (result.keypoints.conf.cpu().numpy()
                   if result.keypoints.conf is not None
                   else np.ones(kpts_xy.shape[:2]))
    else:
        kpts_xy = np.zeros((len(raw_boxes), 17, 2))
        kpts_cf = np.zeros((len(raw_boxes), 17))

    detections = [
        (0, float(c), tuple(int(v) for v in b))
        for b, c in zip(raw_boxes, raw_confs)
    ]
    tracked_objects = track_manager.update(source_id, detections)

    people = []
    for t in tracked_objects:
        best_iou, best_i = 0.0, None
        for i, b in enumerate(raw_boxes):
            iou = _box_iou(t.bbox, b)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i is None or best_iou < POSE_MATCH_IOU:
            continue
        people.append((t.track_id, kpts_xy[best_i], kpts_cf[best_i],
                       np.array(t.bbox, dtype=float)))

    return people


# ============================================================
# ENGINE LOADING  (same as test_fall_det.py)
# ============================================================

def _batch_capacity(batch_file):
    try:
        with open(batch_file) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def ensure_pose_engine():
    capacity = _batch_capacity(POSE_ENGINE_BATCH_FILE) if os.path.exists(POSE_ENGINE_PATH) else None
    if capacity is not None and capacity >= POSE_ENGINE_MAX_BATCH:
        print(f"Pose engine ready: {POSE_ENGINE_PATH}  (batch={capacity})")
        return POSE_ENGINE_PATH

    if os.path.exists(POSE_ENGINE_PATH):
        print(f"Pose engine batch capacity {capacity} < {POSE_ENGINE_MAX_BATCH} - rebuilding...")
    else:
        print(f"Exporting {POSE_WEIGHTS_PT} -> {POSE_ENGINE_PATH} "
              f"(batch={POSE_ENGINE_MAX_BATCH}, imgsz={POSE_IMGSZ}) - one-time, ~2-3 min...")

    if not os.path.exists(POSE_WEIGHTS_PT):
        raise SystemExit(
            f"[ERROR] Pose weights not found: {POSE_WEIGHTS_PT}\n"
            f"Place yolo26m-pose.pt in AI_inferencing/models/ and re-run."
        )
    YOLO(POSE_WEIGHTS_PT).export(
        format="engine", half=True, dynamic=True,
        batch=POSE_ENGINE_MAX_BATCH, imgsz=POSE_IMGSZ, device=0, verbose=False,
    )
    with open(POSE_ENGINE_BATCH_FILE, "w") as f:
        f.write(str(POSE_ENGINE_MAX_BATCH))
    print(f"Pose engine exported: {POSE_ENGINE_PATH}")
    return POSE_ENGINE_PATH


# ============================================================
# PARSE SOURCES  (same as test_fall_det.py)
# ============================================================

def parse_sources(spec):
    """Parse CAMERAS env var into [(source_id, source, kind)] list.

    Each token can be:
      cam-r01            -> MediaMTX camera name  (uses CameraStream)
      rtsp://host/path    -> RTSP URL              (uses CameraStream)
      /path/to/file.mp4  -> local video file       (uses cv2.VideoCapture)
    """
    sources = []
    for i, tok in enumerate(t.strip() for t in spec.split(",")):
        if not tok:
            continue
        if "://" in tok:
            sid = tok.rstrip("/").split("/")[-1].upper() or f"SRC{i}"
            sources.append((sid, tok, "rtsp"))
        elif os.path.isfile(tok):
            sid = os.path.splitext(os.path.basename(tok))[0].upper()
            sources.append((sid, tok, "file"))
        else:
            url = f"{MEDIAMTX_BASE.rstrip('/')}/{tok}"
            sources.append((tok.upper(), url, "rtsp"))
    return sources


# ============================================================
# MAIN
# ============================================================

def main():
    source_list = parse_sources(CAMERAS)
    if not source_list:
        raise SystemExit("No sources. Set CAMERAS='cam-r01,cam-r09' or a comma-separated list.")

    print(f"\nPose-Based Fight + Fall Detection  |  {len(source_list)} source(s)")
    print(f"Engine : {POSE_ENGINE_PATH}")
    print(f"FPS    : {TARGET_FPS}  |  imgsz: {POSE_IMGSZ}")
    print(f"Fall   : transition-required={FALL_REQUIRE_TRANSITION}  "
          f"state_hz={FALL_STATE_HZ} (every {FALL_STATE_STRIDE} frame(s))  "
          f"window={FALL_STATE_SAMPLES / FALL_STATE_HZ:.0f}s")
    print(f"Fight  : mutual={FIGHT_MUTUAL}  "
          f"min_strikes_each={FIGHT_MIN_STRIKES_EACH if FIGHT_MUTUAL else FIGHT_MIN_VIOLENT}  "
          f"min_kpts={FIGHT_MIN_KPTS}")
    print(f"Sources: {[s[0] for s in source_list]}\n")

    model = YOLO(ensure_pose_engine(), task="pose")
    print("Pose model loaded.\n")

    # One shared tracker for every source - TrackManager is already keyed
    # internally on (source_id, group), same as the rest of this project's
    # test scripts.
    track_manager = tracking.TrackManager(
        person_classes={0},
        vehicle_classes=set(),
        class_names={0: "person"},
        max_disappeared=TRACK_MAX_DISAPPEARED,
        dist_thresh=TRACK_DIST_THRESH,
        iou_thresh=TRACK_IOU_THRESH,
        duplicate_iou_thresh=TRACK_DUPLICATE_IOU_THRESH,
    )
    print(
        f"Tracker: max_disappeared={TRACK_MAX_DISAPPEARED}  min_hits={tracking.MIN_HITS}  "
        f"dist_thresh={TRACK_DIST_THRESH}  iou_thresh={TRACK_IOU_THRESH}  "
        f"duplicate_iou={TRACK_DUPLICATE_IOU_THRESH}"
        "   (finalized in test_track.py, same as the rest of this project's test scripts)\n"
    )

    streams = []
    file_caps = {}

    for sid, url, kind in source_list:
        if kind == "rtsp":
            print(f"Starting stream  {sid}  {url}")
            cam = CameraStream(camera_id=sid, source=url, queue_size=3, loop_video=False)
            cam.start()
            streams.append((sid, cam))
        else:
            print(f"Opening file     {sid}  {url}")
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print(f"  [WARN] Cannot open {url} - skipping")
                continue
            file_caps[sid] = cap

    print("\nRunning. Press q / Esc to quit.\n")

    ALERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fight/fall frames will be saved to: {ALERT_OUTPUT_DIR}\n")

    # Per-source state: fall state (per-track SUBSAMPLED posture-state queue +
    # transition latch, ported from fightfalldetection.py's re-tuned version;
    # anonymous fallback keeps the old windowed geometry - it cannot reason
    # about a transition without a stable id to hold history against), plus
    # per-pair two-sided fight-strike history.
    state = {
        sid: {
            "state_hist": defaultdict(lambda: deque(maxlen=FALL_STATE_SAMPLES)),
            "nonlying_run": defaultdict(int),
            "fall_latched": defaultdict(bool),
            "anon_fall_history": deque(maxlen=FALL_WINDOW),
            "anon_fall_hold": 0,
            "pair_a_hits": defaultdict(lambda: deque(maxlen=FIGHT_WINDOW_FRAMES)),
            "pair_b_hits": defaultdict(lambda: deque(maxlen=FIGHT_WINDOW_FRAMES)),
            "prev_kpts": {},
            "prev_confs": {},
            "fight_cooldown": 0,
            # Pairs/tracks already saved for the CURRENT ongoing fight/fall -
            # cleared once a pair/track drops out of confirmed_fight_ids/the
            # confirmed-fall set, so a later, separate event involving the
            # same person(s) gets saved again instead of being silently
            # skipped forever.
            "saved_fight_pairs": set(),
            "saved_fall_tracks": set(),
        }
        for sid, *_ in source_list
    }
    saved_frame_count = 0

    last_frame_id = {sid: None for sid, *_ in source_list}
    last_processed_time = {sid: 0.0 for sid, *_ in source_list}
    # Per-source frame counter the fall-state subsample stride is measured
    # against (every FALL_STATE_STRIDE-th processed frame for that source).
    frame_idx = {sid: 0 for sid, *_ in source_list}

    total_frames = 0
    fall_events = 0
    fight_events = 0
    stats_start = time.perf_counter()

    try:
        while True:
            frames_processed = False
            current_time = time.perf_counter()

            pending = []

            for sid, cam in streams:
                packet = cam.get_latest_frame()
                if packet is None or packet["frame_id"] == last_frame_id[sid]:
                    continue
                if (current_time - last_processed_time[sid]) < FRAME_INTERVAL:
                    continue
                last_frame_id[sid] = packet["frame_id"]
                last_processed_time[sid] = current_time
                pending.append((sid, packet["frame"]))

            for sid, cap in file_caps.items():
                if (current_time - last_processed_time[sid]) < FRAME_INTERVAL:
                    continue
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                last_processed_time[sid] = current_time
                pending.append((sid, frame))

            for sid, frame in pending:
                frames_processed = True
                total_frames += 1
                frame_idx[sid] += 1
                st = state[sid]

                result = model.predict(
                    frame, conf=PERSON_CONF, imgsz=IMG_SIZE, verbose=False, classes=[0],
                )[0]

                people = build_people(track_manager, sid, result)

                # --- per-person limb velocities (body-heights / second) -----
                limb_vel = {}
                for track_id, kpts, confs, bbox in people:
                    limb_vel[track_id] = limb_velocities(
                        st["prev_kpts"].get(track_id), st["prev_confs"].get(track_id),
                        kpts, confs, bbox, TARGET_FPS)

                # --- fight detection: close pairs -> windowed confirmation --
                # Two-sided (FIGHT_MUTUAL): both people must land strikes on
                # each other within the window - a genuine exchange, not one
                # person's fast-limb reading (crowd key-point bleed, a wave,
                # a stumble) counted alone. See FIGHT_MUTUAL's comment above.
                close_pairs = detect_fights(people, limb_vel)
                confirmed_fight_ids = set()
                fight_pairs = []
                for ta, tb, ca_c, cb_c, a_hits_b, b_hits_a in close_pairs:
                    # Orient the per-side counters consistently with the sorted key.
                    if ta <= tb:
                        key, fh, sh = (ta, tb), a_hits_b, b_hits_a
                    else:
                        key, fh, sh = (tb, ta), b_hits_a, a_hits_b
                    st["pair_a_hits"][key].append(bool(fh))
                    st["pair_b_hits"][key].append(bool(sh))
                    hits_a = sum(st["pair_a_hits"][key])
                    hits_b = sum(st["pair_b_hits"][key])
                    if FIGHT_MUTUAL:
                        confirmed = (hits_a >= FIGHT_MIN_STRIKES_EACH and
                                    hits_b >= FIGHT_MIN_STRIKES_EACH)
                    else:
                        confirmed = (hits_a + hits_b) >= FIGHT_MIN_VIOLENT
                    if confirmed:
                        confirmed_fight_ids.add(ta)
                        confirmed_fight_ids.add(tb)
                        fight_pairs.append((ta, tb, ca_c, cb_c))

                # --- fall + drawing ------------------------------------------
                # Tracked people: SUBSAMPLED posture-state queue + transition
                # latch (ported from fightfalldetection.py's re-tuned version -
                # see FALL_REQUIRE_TRANSITION's comment above for why this
                # replaced a flat windowed-geometry hold). Untracked people
                # (no stable id - see build_people()) fall back to the old
                # windowed geometry unchanged, since a transition needs history
                # held against a specific person.
                any_fall = False
                confirmed_fall_ids = set()
                for track_id, kpts, confs, bbox in people:
                    info = analyze_posture(kpts, confs, bbox)
                    raw_fall = info["is_fall"]

                    if track_id is not None:
                        # Subsample: record a state only every
                        # FALL_STATE_STRIDE-th processed frame for this source
                        # (FALL_STATE_HZ per second) and update the
                        # clear-counter there.
                        if frame_idx[sid] % FALL_STATE_STRIDE == 0:
                            p_state = posture_state(info)
                            st["state_hist"][track_id].append(p_state)
                            if p_state == "lying":
                                st["nonlying_run"][track_id] = 0
                            else:
                                st["nonlying_run"][track_id] += 1
                                if st["nonlying_run"][track_id] >= FALL_CLEAR_SAMPLES:
                                    st["fall_latched"][track_id] = False

                        q = list(st["state_hist"][track_id])
                        # "now lying": most of the last few samples are 'lying'.
                        recent = q[-FALL_RECENT_SAMPLES:]
                        now_lying = recent.count("lying") >= FALL_MIN_LYING
                        # "was standing": a 'standing' sample exists anywhere in
                        # the (several-second) history -> a real upright->ground
                        # transition, not someone who was always down.
                        was_standing = "standing" in q

                        if now_lying and (was_standing or not FALL_REQUIRE_TRANSITION):
                            st["fall_latched"][track_id] = True
                        confirmed_fall = st["fall_latched"][track_id]
                    else:
                        st["anon_fall_history"].append(raw_fall)
                        down_recent = sum(st["anon_fall_history"]) >= FALL_MIN_DOWN
                        if down_recent:
                            st["anon_fall_hold"] = FALL_HOLD_FRAMES
                        else:
                            st["anon_fall_hold"] = max(0, st["anon_fall_hold"] - 1)
                        confirmed_fall = st["anon_fall_hold"] > 0

                    if confirmed_fall:
                        any_fall = True
                        if track_id is not None:
                            confirmed_fall_ids.add(track_id)

                    in_fight = track_id in confirmed_fight_ids
                    label, color = decide_display(info, confirmed_fall, in_fight)
                    draw_person(frame, kpts, confs, bbox, label, color, track_id)

                # --- alarms ---------------------------------------------------
                if confirmed_fight_ids:
                    st["fight_cooldown"] = FIGHT_COOLDOWN_FRAMES
                else:
                    st["fight_cooldown"] = max(0, st["fight_cooldown"] - 1)

                if confirmed_fight_ids or st["fight_cooldown"] > 0:
                    draw_fight_links(frame, fight_pairs)

                banners = []
                if any_fall:
                    banners.append("FALL DETECTED")
                    fall_events += 1
                    print(f"[FALL DETECTED]  {sid}  frame={total_frames}")
                if confirmed_fight_ids or st["fight_cooldown"] > 0:
                    banners.append("FIGHT / VIOLENCE DETECTED")
                    if confirmed_fight_ids:
                        fight_events += 1
                        print(f"[FIGHT DETECTED]  {sid}  frame={total_frames}  "
                              f"pairs={sorted(confirmed_fight_ids)}")
                draw_alert_banner(frame, banners)

                # --- save fully-annotated frames for newly-triggered fights -
                # One save per pair per fight EPISODE, not one per frame of an
                # ongoing fight: fight_pairs already only holds pairs confirmed
                # THIS frame, so diffing against what was already saved for the
                # current episode gives a rising-edge trigger. Saved AFTER all
                # drawing (skeletons, boxes, fight links, alert banner) and
                # BEFORE the display-only resize, so the file keeps full
                # source resolution with every annotation on it.
                current_pair_keys = {(min(ta, tb), max(ta, tb)) for ta, tb, *_ in fight_pairs}
                for pair_key in current_pair_keys - st["saved_fight_pairs"]:
                    out_path = ALERT_OUTPUT_DIR / f"{sid}_{pair_key[0]}_{pair_key[1]}_{int(time.time())}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    saved_frame_count += 1
                    print(f"[SAVED] {out_path}")
                st["saved_fight_pairs"] = current_pair_keys

                # --- save fully-annotated frames for newly-triggered falls --
                # Same rising-edge logic as the fight save above, per track
                # instead of per pair, same shared ALERT_OUTPUT_DIR.
                for fall_track_id in confirmed_fall_ids - st["saved_fall_tracks"]:
                    out_path = ALERT_OUTPUT_DIR / f"{sid}_fall_{fall_track_id}_{int(time.time())}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    saved_frame_count += 1
                    print(f"[SAVED] {out_path}")
                st["saved_fall_tracks"] = confirmed_fall_ids

                # --- remember this frame's key-points for next-frame motion -
                st["prev_kpts"] = {}
                st["prev_confs"] = {}
                for track_id, kpts, confs, _b in people:
                    st["prev_kpts"][track_id] = kpts
                    st["prev_confs"][track_id] = confs

                disp = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                cv2.imshow(f"Fight + Fall Detection  |  {sid}", disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if not frames_processed:
                time.sleep(0.003)

            elapsed = time.perf_counter() - stats_start
            if elapsed >= 5.0:
                n = len(streams) + len(file_caps)
                print(
                    f"[Heartbeat] FPS: {total_frames / elapsed:.1f}"
                    f"  ({total_frames / elapsed / max(n, 1):.1f}/source)"
                    f"  sources: {n}"
                    f"  active_tracks: {track_manager.active_track_count()}"
                    f"  falls: {fall_events}  fights: {fight_events}"
                    f"  saved_frames: {saved_frame_count}"
                )
                total_frames = 0
                stats_start = time.perf_counter()

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        for _, cam in streams:
            cam.stop()
        for cap in file_caps.values():
            cap.release()
        cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print(f"Shutdown complete. Saved {saved_frame_count} fight/fall frame(s) to {ALERT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
