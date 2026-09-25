"""
Track management - stable track IDs on top of JeztSort.

jeztsort.ImprovedTracker is class-agnostic and numbers its tracks from 0 within
each instance. Two layers are added here; JeztSort itself is not modified.

  * One tracker per (camera, object group). Groups are "person", "vehicle" and
    "object" (carried bag/box, for the abandoned-object analytic). Keeping
    them apart means a car can never inherit a person's ID when their boxes
    overlap.

  * A process-wide label allocator. Two cameras would otherwise both start at
    track 0 and both report "P-0001". Each internal tracker id is mapped once,
    on first sight, to a globally unique label in the dashboard's documented
    format (P-0017 / V-0032), and that label stays with the track for its life.

Only CONFIRMED tracks are returned - JeztSort confirms a track after min_hits
consecutive matches, which drops single-frame false positives before they can
become events.

TUNING - the five knobs below (each documented in detail right above its
constant) are how you fix id churn, lost tracks, and duplicate boxes for the
same object. Quick lookup by SYMPTOM:

  same person gets a new id after being briefly hidden   -> raise MAX_DISAPPEARED
  same person gets a new id while continuously visible   -> raise DIST_THRESH (the most
                                                              likely cause now that AI runs
                                                              on the 1080p main stream -
                                                              see its note below)
  a shadow / reflection briefly becomes a fake track      -> raise MIN_HITS
  two people close together keep swapping ids             -> lower DIST_THRESH / IOU_THRESH
  a fast-moving person is never tracked at all             -> lower MIN_HITS, raise IOU/DIST
  ONE person shows TWO boxes AT THE SAME TIME              -> lower DUPLICATE_IOU_THRESH (a
                                                              same-frame problem, not a
                                                              frame-to-frame one - see its
                                                              note below for why)

TrackManager accepts each of these as a constructor argument
(max_disappeared=/min_hits=/iou_thresh=/dist_thresh=/duplicate_iou_thresh=) to
tune ONE instance without touching the module default - test_detect.py does
this, driven by env vars, so you can experiment live without editing this
file. A TrackManager that does not pass an override uses the module default
below; multicam_inf.py (production) does not override, so retuning a default
here changes production behaviour the next time it restarts.

Module-default env overrides (see each constant below for what it controls):
  TRACK_MAX_DISAPPEARED     frames a track survives unmatched     (default 25)
  TRACK_MIN_HITS            matches before a track is confirmed   (default 3)
  TRACK_IOU_THRESH          IoU for first-pass matching           (default 0.3)
  TRACK_DIST_THRESH         centroid px for second-pass match     (default 140)
  TRACK_DUPLICATE_IOU_THRESH  same-frame duplicate-box suppression (default 0.55)
"""

import itertools
import os
import random
import threading
import time

from jeztsort import ImprovedTracker, iou


# ============================================================
# CONFIGURATION
#
# Read once as the module loads; each becomes the DEFAULT for every
# TrackManager that does not override it explicitly (see TrackManager.__init__
# below) - which includes multicam_inf.py, production.
# ============================================================

# Frames a track survives with NO matching detection before JeztSort drops it.
#   TOO LOW  -> a person briefly occluded (a pole, a doorway, another person
#               crossing in front of them) loses their track. When they
#               reappear they get a BRAND NEW id - this is exactly "missing
#               tracking, multiple ids for the same person."
#   TOO HIGH -> a track that has actually left the scene keeps "existing" for
#               longer than it should, so a different person/vehicle walking
#               through the same spot soon after can occasionally re-attach
#               to the stale track instead of starting a clean new one.
# Counted in INFERENCE frames processed, NOT camera frames or wall-clock time -
# its real-world meaning depends entirely on how fast TrackManager.update() is
# being called. At multicam_inf.py's production TARGET_INFERENCE_FPS of 5,
# 25 frames is ~5s of grace; at test_detect.py's default 10 FPS it is ~2.5s.
# Raising TARGET_INFERENCE_FPS shrinks this in wall-clock terms unless you
# raise this too.
MAX_DISAPPEARED = int(os.getenv("TRACK_MAX_DISAPPEARED", "300"))

# Consecutive matched frames required before a track is CONFIRMED at all (and
# gets a P-/V-/O- id, and starts being returned by update()).
#   TOO LOW  -> a single-frame false detection (motion blur, a shadow, a
#               reflection) gets promoted to a real, briefly-flickering track.
#   TOO HIGH -> someone walking quickly through the frame - or a track that
#               keeps narrowly missing a match - can leave before ever being
#               confirmed, i.e. "missing tracking" for fast-moving objects.
MIN_HITS = int(os.getenv("TRACK_MIN_HITS", "3"))

# Minimum box overlap (IoU, 0.0-1.0) between this frame and the track's last
# known box to call it the SAME object on the first, cheap matching pass.
#   TOO LOW  -> two different people standing close together can swap ids, or
#               a person can "steal" a nearby vehicle's track.
#   TOO HIGH -> a person moving briskly, combined with a low inference rate
#               (few frames per second), moves further between frames than
#               the overlap threshold allows. The first pass then fails and
#               either falls through to the coarser DIST_THRESH pass below,
#               or loses the match entirely and starts a new id.
IOU_THRESH = float(os.getenv("TRACK_IOU_THRESH", "0.35"))

# Centroid distance, in RAW PIXELS of the frame the model runs on, allowed for
# the second (fallback) matching pass used when the IoU pass above fails.
#   TOO LOW  -> the single most likely cause of "same person gets a new id
#               while clearly still on screen." If a person's box centre
#               moves more pixels than this between two consecutive inference
#               frames, the tracker gives up and starts a fresh id.
#   TOO HIGH -> two different people (or vehicles) passing close to each
#               other can get merged onto the same id.
# THIS IS RESOLUTION-DEPENDENT and must be retuned if the source resolution
# changes. AI inference now runs on the 1920x1080 MAIN stream (previously the
# 704x576 substream) - the same real-world walking speed now covers roughly
# 2.7x more pixels per frame (1920/704), so a value tuned for the old
# substream is about 2.7x too tight here. 140 is that old default (50) scaled
# for the current resolution - a starting point, not a proven-optimal number;
# watch the HUD's "churn" figure in test_detect.py while nudging it. If you
# point a script at a lower-resolution source (a substream, a different
# camera), scale this back down proportionally.
DIST_THRESH = float(os.getenv("TRACK_DIST_THRESH", "150"))

# Consecutive UNMATCHED inference frames after which a re-match is treated as a
# DIFFERENT physical object, and the exported label is retired in favour of a
# new one. This is an identity rule layered ON TOP of JeztSort; it does not
# change MAX_DISAPPEARED, the tracker, or any matching threshold.
#
# WHY THIS EXISTS - the track-collision defect.
#   MAX_DISAPPEARED is 300 inference frames, and multicam_inf.py runs at
#   TARGET_INFERENCE_FPS = 10, so a track with NO matching detection survives
#   for 30 SECONDS with its box frozen where the vehicle was last seen. When a
#   match finally arrives, jeztsort.py's update() sets `t.disappeared = 0` and
#   the track simply continues - the same internal id, and therefore the same
#   exported label.
#
#   At an entry lane that is exactly what happens: a vehicle leaves, its frozen
#   box sits over the spot the NEXT vehicle stops at, and the next vehicle -
#   well inside DIST_THRESH of that frozen box - is matched to the departed
#   vehicle's track. Every re-match resets the clock, so the chain never
#   breaks. Measured on preserved evidence: 29 of 367 track ids (7.9%) span
#   more than 120s, V-WLOU-0748 held SIX different vehicles over 55 minutes
#   under one id, and 16 of 65 published plate identities (24.6%) sat on such
#   a track. The OCR was right in each case; the ASSOCIATION was wrong.
#
#   The tracker is not at fault and is not modified: coasting is what lets a
#   briefly occluded vehicle keep its id. What was missing is any notion that
#   a long enough gap breaks physical continuity.
#
# 15 frames = 1.5s at the production 10 FPS.
#   TOO LOW  -> a vehicle occluded by a passing lorry comes back as a new id
#               (fragmentation: one vehicle, several ids).
#   TOO HIGH -> the collision above returns (two vehicles, one id).
# The asymmetry is deliberate. Fragmentation costs recall on one vehicle;
# collision attributes one vehicle's PLATE to another, which on an evidence
# system is the far worse failure. A vehicle still physically present is still
# being DETECTED, so its `disappeared` stays at 0 and this rule never fires on
# it; firing requires the detector to have lost the object for 1.5s, by which
# time a re-match at the same pixels is more likely a new arrival than the
# original.
MAX_COAST_FRAMES = int(os.getenv("TRACK_MAX_COAST_FRAMES", "15"))

# The same rule, in SECONDS - which is what it always meant.
#
# WHY THIS REPLACED THE FRAME COUNT. "15 frames = 1.5s" is only true while the
# camera is actually inferred at TARGET_INFERENCE_FPS. Measured on this
# deployment, it is not: CAM-R06 and CAM-R12 lose RTP packets, their decoders
# then deliver frames in bursts, and the inference loop - which takes only the
# newest queued frame per visit - reached 1.75 and 1.85 frames per second while
# the eight healthy cameras ran at 7.1-8.2. At 1.75 fps this rule's window was
# 8.6 SECONDS, not 1.5: long enough for a departed vehicle's frozen box to
# capture the next arrival, which is the exact track-collision defect the rule
# exists to prevent. The frame count silently became the wrong rule at the one
# camera that matters most - the ANPR entry lane.
#
# Seconds are measured from the update() call times, so the window is 1.5s at
# 10 fps, at 1.75 fps, and at any rate in between. At the intended 10 fps this
# is 15 frames: the behaviour the frame count was chosen to give.
MAX_COAST_SECONDS = float(os.getenv("TRACK_MAX_COAST_SECONDS", "1.5"))

# The inference rate a frame-count threshold is converted at when nothing has
# been measured yet (the first frames after a camera starts). Matches
# multicam_inf.TARGET_INFERENCE_FPS.
NOMINAL_FPS = float(os.getenv("TRACK_NOMINAL_FPS", "10"))

# An interval between two updates of one camera longer than this is a STALL:
# the camera delivered nothing, so nothing about any vehicle was observed.
#
# WHY THIS EXISTS - measured 2026-09-15 on CAM-R06. Car K49773 sat at the gate
# for 21 seconds, detected at 0.92-0.96 on every frame with a pixel-stable box,
# and was issued THREE track ids and three ANPR jobs. The stream was stalling
# under RTP loss: the recording has 2s holes at 18:12:42-46 and no frames at
# all from 18:12:58.7 to 18:13:12.8. Counting wall time, the stopped car was
# "unmatched" for the whole stall, so the 1.5s identity rule retired its label
# the moment frames resumed (18:13:09.1 - the new id's exact birth), and the
# long intervals inflated the measured frame spacing until JeztSort's window
# shrank to a couple of frames and one low-confidence frame dropped the track.
#
# A stall is ABSENCE OF EVIDENCE, not evidence the vehicle left. So a stall
# advances the identity clock by one ordinary frame interval, and is kept out
# of the rate estimate. Only frames that ARRIVED without matching a track count
# towards retiring it - which is what the frame-count rule always did, and what
# made it stall-proof.
STALL_SECONDS = float(os.getenv("TRACK_STALL_SECONDS", "1.0"))

# Same-frame duplicate suppression. Two detections in ONE frame whose IoU is
# at least this is almost certainly the SAME physical object seen twice by
# the detector - a second, slightly different box around the same person or
# vehicle (more likely at 1080p, where the model has more room to produce two
# candidate boxes at slightly different scales/crops for one object).
#
# WHY THIS EXISTS: JeztSort matches each existing track to AT MOST ONE
# detection per frame, but ANY detection left over afterwards is registered
# as a BRAND NEW track with no further checks (see jeztsort.py's
# ImprovedTracker.update() - its final "register any remaining detections"
# step). If the detector hands it two overlapping boxes for one person, one
# binds to that person's existing track and the OTHER becomes a second,
# independent track - which, if the duplicate keeps recurring frame after
# frame, gets CONFIRMED just like a real object. This is what produces "one
# person, two bounding boxes": it is a SAME-FRAME duplicate problem, not a
# frame-to-frame continuity problem, so none of the four knobs above fix it -
# this filter runs BEFORE detections ever reach JeztSort, so the "loser" box
# of a duplicate pair never gets the chance to seed a track at all.
#   TOO LOW  -> two genuinely different people/vehicles standing close
#               together can have the lower-confidence one silently dropped.
#   TOO HIGH -> stops suppressing almost nothing; duplicates reach the
#               tracker again and can seed a second, confirmed ghost track.
# Only ever compares detections of the SAME class (a car box and a truck box
# are never merged, even if they overlap heavily) and always keeps the
# higher-confidence box of the pair.
DUPLICATE_IOU_THRESH = float(os.getenv("TRACK_DUPLICATE_IOU_THRESH", "0.30"))


# ── VEHICLE-ONLY OVERRIDES ──────────────────────────────────────────────────
# The five knobs above are shared by every group. These four replace them for
# the VEHICLE tracker alone, and each defaults to the shared value, so a
# deployment that sets none of them behaves exactly as before.
#
# WHY VEHICLES NEED THEIR OWN NUMBERS
#   The shared values were tuned watching PEOPLE walk. A vehicle is a
#   different tracking problem on the same frames: it is larger, it moves
#   several times faster, and at an entry lane it alternates between stopped
#   and accelerating. Between two inference frames at 10 FPS a car covers far
#   more of its own box length than a walker does, so the frame-to-frame
#   overlap that identifies it is systematically lower - the IoU pass fails,
#   the match falls through to the coarser distance pass or is lost outright,
#   and the same car collects a second V- id.
#
#   Person tracking is NOT touched by any of this: group_for() sends person
#   classes to their own ImprovedTracker, built from the shared constants.
#
# WHAT IS DELIBERATELY *NOT* HERE
#   MAX_COAST_FRAMES stays global and stays at 15. It is the rule that stops
#   one id spanning two physical vehicles, and it is the reason a vehicle gone
#   for more than 1.5s comes back as a NEW logical track. Loosening
#   association is safe precisely because that boundary is untouched.
# Frames a VEHICLE track survives unmatched, against 300 shared.
#
# 300 frames is 30s at the production 10 FPS, and for a vehicle that is not
# grace, it is a graveyard: every vehicle that has left the frame keeps a
# frozen box in the tracker for half a minute. JeztSort matches tracks in
# insertion order (jeztsort.py's `for i in range(len(track_boxes))` over an
# OrderedDict), so those OLD frozen boxes get first pick of this frame's
# detections and can take one that belongs to a live vehicle. The live vehicle
# then goes unmatched, its `disappeared` climbs, and it is the CORRECT track
# that ends up losing its identity. Measured on 1,200 recorded frames of two
# vehicle scenes, dropping this to 20 cut ids-per-vehicle from 2.44 to 1.63
# and id switches from 167 to 74 - and cut false merges as well, because a
# stale box that no longer exists cannot capture anybody.
#
# WHY 20 AND NOT 15. 15 scores slightly better still, but it is exactly
# MAX_COAST_FRAMES, and the identity rule below fires on `gap > 15`. A track
# deregistered at 15 can never BE re-matched with a gap of 16, so the rule
# would never run, `generations` would stop counting, and the protection
# against two vehicles sharing one id would quietly become dead code. 20
# leaves the rule a live window (gaps of 16-20 retire the label explicitly)
# and gives up almost nothing: 1.63 ids per vehicle either way.
#
# A gap longer than 20 is still safe: the track is deregistered, the next
# vehicle registers a fresh one, and it gets a new label. Both paths end in a
# new logical identity, which is the invariant that matters.
VEHICLE_MAX_DISAPPEARED = int(os.getenv("TRACK_VEHICLE_MAX_DISAPPEARED", "30"))

# The same two windows expressed in seconds. JeztSort counts frames, so
# TrackManager converts these with each camera's MEASURED update rate before
# every update: a starved camera keeps the intended 3s of coasting instead of
# 30 frames that silently became 17s. Set either to 0 to keep the raw frame
# counts above (the pre-2026-09-14 behaviour).
VEHICLE_MAX_DISAPPEARED_SECONDS = float(
    os.getenv("TRACK_VEHICLE_MAX_DISAPPEARED_SECONDS", "3.0"))
MAX_DISAPPEARED_SECONDS = float(
    os.getenv("TRACK_MAX_DISAPPEARED_SECONDS", "30.0"))

# Deliberately inherits MIN_HITS (3). Sweeping it moved nothing worth having:
# 2 gained 0.02 ids per vehicle, 4 lost 2 points of clean tracks. The knob
# exists so vehicles CAN be confirmed differently without touching people.
VEHICLE_MIN_HITS = int(os.getenv("TRACK_VEHICLE_MIN_HITS", str(MIN_HITS)))

# First-pass overlap for vehicles, against 0.35 shared.
#
# A car at 10 FPS moves a large fraction of its own box between frames, so the
# overlap that identifies it is systematically lower than a walker's. At 0.35
# the IoU pass misses and the match falls through to the centroid pass, which
# is the pass that confuses NEIGHBOURING vehicles - so a gate that is too
# tight does not prevent merges, it CAUSES them.
#
# 0.30, not the 0.25 that scores best on open road. The two scene types
# disagree, and the disagreement is the whole point:
#
#   open road (CAM-R07/R08)  0.25 is clearly better - 1.41 vs 1.66 ids per
#                            vehicle. Vehicles are separated, so a loose gate
#                            has nothing wrong to grab.
#   entry lane (CAM-R06)     0.25 is WORSE than doing nothing - false merges
#                            20 -> 21 and clean tracks 75.0% -> 71.9%. Queued
#                            cars stand bumper to bumper, so a gate that loose
#                            starts matching a car to the one in front.
#
# The entry lane is the ANPR camera, and a merge there hands one vehicle's
# PLATE to another - the worst failure this system can make. 0.30 is the value
# that is better than the current behaviour on BOTH: it beats baseline on
# every measure on all three recordings and loses to baseline on none.
VEHICLE_IOU_THRESH = float(os.getenv("TRACK_VEHICLE_IOU_THRESH", "0.30"))

# Deliberately inherits DIST_THRESH (150) - NOT loosened. This is the one
# parameter where the brief's warning is measurably true: raising it to 200
# cut fragmentation a little and pushed false merges from 79 to 92, because
# centroid matching is what lets one vehicle capture its neighbour's box.
# Tightening it to 130 does help (merges 61 -> 54), but it is resolution-
# dependent and was not part of this change, so it is left alone and noted
# here as the next thing to try if merges need to come down further.
VEHICLE_DIST_THRESH = float(os.getenv("TRACK_VEHICLE_DIST_THRESH", str(DIST_THRESH)))


PERSON_GROUP = "person"
VEHICLE_GROUP = "vehicle"
OBJECT_GROUP = "object"          # carried bag/box, tracked for abandoned-object

GROUP_PREFIX = {
    PERSON_GROUP: "P",
    VEHICLE_GROUP: "V",
    OBJECT_GROUP: "O",
}


# One counter for the whole process, so labels are unique across every camera.
_COUNTER = itertools.count(1)
_COUNTER_LOCK = threading.Lock()


# ── Run identity ────────────────────────────────────────────────────────────
# The counter above is process-local: it restarts at 1 every time this process
# starts, so a plain "P-0008" is only unique for the lifetime of one run. The
# dashboard groups a track's events by (camera_id, track_id), and without a run
# component that grouping silently merges two different people who happened to
# be the eighth track of their respective runs. This was observable in the live
# database as a single "track" holding events twenty hours apart.
#
# A short run id makes the exported label globally unique. It is generated once
# at import and never changes for the life of the process; a restart produces a
# different one. Only the EXPORTED label carries it - JeztSort's internal ids
# are untouched, and nothing downstream parses the label beyond its P-/V-/O-
# prefix.

_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _base36(value, width):
    """Fixed-width base36, least significant digits kept."""
    out = ""
    for _ in range(width):
        value, remainder = divmod(value, 36)
        out = _BASE36[remainder] + out
    return out


def _make_run_id():
    """
    Four characters: three from the clock, one random.

    The clock part keeps run ids roughly time-ordered and wraps only every
    ~13 hours; the random character separates two processes that start within
    the same second. Short enough to stay readable in the UI and to keep the
    whole label inside the dashboard's 24-character track_id column.
    """
    clock = _base36(int(time.time()), 3)
    salt = _base36(random.randrange(36), 1)
    return clock + salt


# Overridable so tests can pin it and so an operator can correlate a run.
RUN_ID = os.getenv("TRACK_RUN_ID") or _make_run_id()


def _next_label(group):
    with _COUNTER_LOCK:
        number = next(_COUNTER)

    return f"{GROUP_PREFIX[group]}-{RUN_ID}-{number:04d}"


# ============================================================
# TRACKED OBJECT
# ============================================================

class TrackedObject:
    """One confirmed track, seen in the frame that was just processed."""

    def __init__(
        self,
        camera_id,
        track_id,
        group,
        class_id,
        class_name,
        confidence,
        bbox,
        is_new,
        raw_track_id=None,
    ):
        self.camera_id = camera_id
        # THREE DISTINCT IDENTITIES - do not confuse them:
        #   raw_track_id  JeztSort's internal integer, per (camera, group)
        #                 tracker instance. Debugging only. NOT unique across
        #                 cameras, NOT stable across a restart, and one raw id
        #                 can span several logical vehicles (that is the whole
        #                 point of the generation rule below).
        #   track_id      the LOGICAL vehicle-track identity, exported and
        #                 persisted. "V-2K7Q-0017" (prefix-run-sequence). One
        #                 of these means one continuous physical observation.
        #   event_id      allocated downstream by the dashboard, per event.
        #                 Independently addressable and never derived from a
        #                 track id - evidence must remain retrievable by it.
        self.track_id = track_id        # "P-2K7Q-0017"  (prefix-run-sequence)
        self.raw_track_id = raw_track_id
        self.group = group              # "person" | "vehicle"
        self.class_id = class_id
        self.class_name = class_name    # "person", "car", ...
        self.confidence = confidence
        self.bbox = bbox                # [x1, y1, x2, y2] ints, frame pixels
        self.is_new = is_new            # first frame this label was emitted

    def __repr__(self):
        return f"<TrackedObject {self.camera_id} {self.track_id} {self.class_name}>"


def _suppress_duplicates(group_detections, iou_thresh):
    """
    Collapse near-identical SAME-CLASS detections within one frame down to
    the single highest-confidence box, before they ever reach JeztSort.

    See DUPLICATE_IOU_THRESH above for why this exists. group_detections is
    a list of (class_id, confidence, box); returns the same shape, filtered.
    """
    ordered = sorted(group_detections, key=lambda d: d[1], reverse=True)

    kept = []

    for class_id, confidence, box in ordered:

        duplicate = any(
            class_id == kept_class_id and iou(box, kept_box) >= iou_thresh
            for kept_class_id, _, kept_box in kept
        )

        if duplicate:
            continue

        kept.append((class_id, confidence, box))

    return kept


# ============================================================
# TRACK MANAGER
# ============================================================

class TrackManager:
    """
    Holds the per-camera, per-group JeztSort instances and the label mapping.

    The class taxonomy is injected rather than defined here, so multicam_inf.py
    stays the single place COCO ids are declared.
    """

    def __init__(self, person_classes, vehicle_classes, class_names,
                 object_classes=None,
                 max_disappeared=None, min_hits=None,
                 iou_thresh=None, dist_thresh=None,
                 duplicate_iou_thresh=None,
                 vehicle_max_disappeared=None, vehicle_min_hits=None,
                 vehicle_iou_thresh=None, vehicle_dist_thresh=None):
        self._person_classes = set(person_classes)
        self._vehicle_classes = set(vehicle_classes)
        # Carried items (bag/box) tracked for the abandoned-object analytic.
        # Optional so existing callers (test_yolo.py) keep working unchanged.
        self._object_classes = set(object_classes or ())
        self._class_names = class_names

        # Per-instance tuning overrides, falling back to the module defaults
        # above (what multicam_inf.py/production uses) when not given. See
        # each constant's comment above for what it controls.
        self._max_disappeared = MAX_DISAPPEARED if max_disappeared is None else max_disappeared
        self._min_hits = MIN_HITS if min_hits is None else min_hits
        self._iou_thresh = IOU_THRESH if iou_thresh is None else iou_thresh
        self._dist_thresh = DIST_THRESH if dist_thresh is None else dist_thresh
        self._duplicate_iou_thresh = (
            DUPLICATE_IOU_THRESH if duplicate_iou_thresh is None else duplicate_iou_thresh
        )

        # Vehicle-only overrides. Each falls back to the module default, which
        # itself falls back to the shared value - so leaving all of them unset
        # gives the pre-existing single-tracker behaviour exactly.
        self._vehicle_max_disappeared = (
            VEHICLE_MAX_DISAPPEARED if vehicle_max_disappeared is None
            else vehicle_max_disappeared
        )
        self._vehicle_min_hits = (
            VEHICLE_MIN_HITS if vehicle_min_hits is None else vehicle_min_hits
        )
        self._vehicle_iou_thresh = (
            VEHICLE_IOU_THRESH if vehicle_iou_thresh is None else vehicle_iou_thresh
        )
        self._vehicle_dist_thresh = (
            VEHICLE_DIST_THRESH if vehicle_dist_thresh is None else vehicle_dist_thresh
        )

        # (camera_id, group) -> ImprovedTracker
        self._trackers = {}

        # (camera_id, group) -> {internal_track_id: label}
        self._labels = {}

        # (camera_id, group) -> {internal_track_id: disappeared count observed
        # BEFORE the most recent tracker.update(). jeztsort resets a matched
        # track's counter to 0 inside update(), so the gap it was coasting for
        # is only knowable if it is captured first.
        self._coasting = {}

        # (camera_id, group) -> {internal_track_id: update time, in seconds,
        # when this track was last MATCHED}. The identity rule below is a
        # question about elapsed time, so it is answered with time.
        self._last_seen = {}

        # (camera_id, group) -> (time of the previous update, EMA of the
        # interval between updates). The EMA is what converts a window in
        # seconds into the frame count JeztSort counts in.
        self._rate = {}

        # Diagnostics: how many times a label was retired because its track was
        # re-matched after a coast longer than MAX_COAST_SECONDS.
        self.generations = 0

    # ------------------------------------------------------------------ groups
    def group_for(self, class_id):
        """Which event group a COCO class belongs to, or None to ignore it."""
        if class_id in self._person_classes:
            return PERSON_GROUP

        if class_id in self._vehicle_classes:
            return VEHICLE_GROUP

        if class_id in self._object_classes:
            return OBJECT_GROUP

        return None

    # ---------------------------------------------------------------- internal
    def params_for(self, group):
        """
        The four JeztSort settings this group's tracker is built with.

        Vehicles get their own; person and object keep the shared values. One
        place decides, so a new group cannot silently inherit vehicle tuning.
        """
        if group == VEHICLE_GROUP:
            return {
                "max_disappeared": self._vehicle_max_disappeared,
                "min_hits": self._vehicle_min_hits,
                "iou_thresh": self._vehicle_iou_thresh,
                "dist_thresh": self._vehicle_dist_thresh,
            }

        return {
            "max_disappeared": self._max_disappeared,
            "min_hits": self._min_hits,
            "iou_thresh": self._iou_thresh,
            "dist_thresh": self._dist_thresh,
        }

    def _tracker(self, camera_id, group):
        key = (camera_id, group)

        tracker = self._trackers.get(key)

        if tracker is None:
            tracker = ImprovedTracker(**self.params_for(group))
            self._trackers[key] = tracker
            self._labels[key] = {}
            self._coasting[key] = {}

        return tracker

    # ------------------------------------------------------------------ update
    def _note_update(self, key, now):
        """
        Record this update's time and return (SMOOTHED interval between
        updates, OBSERVED clock) for this camera and group, in seconds.

        The observed clock is wall time with stalls removed (see
        STALL_SECONDS): it is the time the camera was actually watching.

        Smoothed because a single interval is noisy - a bursty stream delivers
        two frames 40ms apart and then nothing for 900ms, and a window sized
        from either number alone would be wrong. The EMA is the rate the camera
        is really being inferred at. None until two updates have been seen.
        """
        previous, average, observed = self._rate.get(key, (None, None, 0.0))
        interval = None if previous is None else max(0.0, now - previous)

        if interval is not None and interval > 0.0:
            if interval <= STALL_SECONDS:
                observed += interval
                average = interval if average is None else (0.8 * average + 0.2 * interval)
            else:
                # A stall: nothing was seen, so it is worth one frame of
                # observation and says nothing about this camera's rate.
                observed += average if average else (1.0 / NOMINAL_FPS)

        self._rate[key] = (now, average, observed)
        return average, observed

    @staticmethod
    def _frames_for(seconds, interval):
        """
        How many frames `seconds` is worth at this measured interval.

        Falls back to NOMINAL_FPS before a rate has been measured, and is
        bounded: never fewer than 2 frames (a window of one frame would
        deregister a track the moment a single detection is missed) and never
        more than 300 (the original shared MAX_DISAPPEARED, so a stalled camera
        cannot grow an unbounded track table).
        """
        rate = (1.0 / interval) if interval else NOMINAL_FPS
        return max(2, min(300, int(round(seconds * rate))))

    def _apply_time_windows(self, tracker, group, interval):
        """
        Give JeztSort the frame count that matches this group's window in
        seconds at the rate this camera is actually achieving.

        JeztSort counts frames and is not modified. Setting the seconds env var
        to 0 leaves the configured frame count alone, which is exactly the
        behaviour before this existed.
        """
        seconds = (VEHICLE_MAX_DISAPPEARED_SECONDS if group == VEHICLE_GROUP
                   else MAX_DISAPPEARED_SECONDS)

        if seconds > 0:
            tracker.max_disappeared = self._frames_for(seconds, interval)

    def update(self, camera_id, detections, now=None):
        """
        Feed one frame's detections in, get this frame's confirmed tracks back.

        detections: list of (class_id, confidence, (x1, y1, x2, y2))
        now:        this frame's time in seconds. Defaults to the monotonic
                    clock, which is what the live loop wants; a replay or a
                    test passes the frame's own time so that a simulated rate
                    is honoured.
        returns:    list of TrackedObject
        """
        now = time.monotonic() if now is None else float(now)
        # Split by group. Every group is updated even when it has no detections
        # this frame, so its tracks age out instead of lingering forever.
        grouped = {PERSON_GROUP: [], VEHICLE_GROUP: [], OBJECT_GROUP: []}

        for class_id, confidence, box in detections:

            group = self.group_for(class_id)

            if group is None:
                continue

            grouped[group].append((class_id, confidence, box))

        tracked = []

        for group, group_detections in grouped.items():

            key = (camera_id, group)

            # Nothing seen and nothing being tracked - skip the bookkeeping.
            if not group_detections and key not in self._trackers:
                continue

            tracker = self._tracker(camera_id, group)

            # How fast this camera is ACTUALLY being inferred, and the frame
            # counts that its windows in seconds are worth at that rate. A
            # camera whose stream stalls gets fewer, longer-spaced frames; its
            # windows have to grow in frames to stay the same in seconds.
            interval, observed = self._note_update(key, now)
            self._apply_time_windows(tracker, group, interval)

            # Collapse same-frame duplicates BEFORE JeztSort ever sees them -
            # an unmatched duplicate would otherwise be registered as a brand
            # new track (see DUPLICATE_IOU_THRESH above).
            group_detections = _suppress_duplicates(group_detections, self._duplicate_iou_thresh)

            boxes = [box for _, _, box in group_detections]

            # Recover class and confidence after tracking: JeztSort carries the
            # box through untouched, so the box identifies its detection.
            detail = {
                box: (class_id, confidence)
                for class_id, confidence, box in group_detections
            }

            # How long each existing track had been coasting BEFORE this
            # frame's matching. jeztsort sets a matched track's `disappeared`
            # to 0 inside update(), so this is the only place the gap can be
            # observed.
            coasting = self._coasting[key]
            before = {
                tid: track.disappeared
                for tid, track in tracker.tracks.items()
            }

            tracks = tracker.update(boxes)

            labels = self._labels[key]
            last_seen = self._last_seen.setdefault(key, {})

            for track in tracks.values():

                if not track.confirmed:
                    continue

                box = tuple(track.box)

                # Not matched this frame (coasting on `disappeared`) - the box
                # is stale, so it must not be reported as a live detection.
                if box not in detail:
                    continue

                class_id, confidence = detail[box]

                # IDENTITY GENERATION RULE.
                # This track has just been matched. If it had been coasting
                # for longer than MAX_COAST_SECONDS, physical continuity with
                # whatever it was tracking before is not credible - a lane
                # that has been empty for 1.5s and then has a vehicle in it
                # is far more likely to hold a NEW vehicle than the departed
                # one. Retire the label; the same raw track goes on, under a
                # new logical identity.
                #
                # Judged on BOTH clocks, and either one is enough to retire.
                # Frames alone stretched to 8.6s on a camera the loop reaches
                # 1.75 times a second (see MAX_COAST_SECONDS); seconds alone
                # would measure nothing where many frames arrive in almost no
                # time - a replay, or the burst intake that hands this loop
                # three frames of one camera in a single tick. Taking the
                # earlier of the two keeps the original guarantee intact and
                # adds the one the frame count silently lost.
                gap = before.get(track.track_id, 0)
                # OBSERVED seconds, so a camera stall is not mistaken for the
                # vehicle having gone (see STALL_SECONDS).
                gap_seconds = observed - last_seen.get(track.track_id, observed)
                stale = gap > MAX_COAST_FRAMES or gap_seconds > MAX_COAST_SECONDS

                if stale and track.track_id in labels:
                    del labels[track.track_id]
                    self.generations += 1

                is_new = track.track_id not in labels

                if is_new:
                    labels[track.track_id] = _next_label(group)

                coasting[track.track_id] = 0
                last_seen[track.track_id] = observed

                tracked.append(
                    TrackedObject(
                        camera_id=camera_id,
                        track_id=labels[track.track_id],
                        group=group,
                        class_id=class_id,
                        class_name=self._class_names.get(class_id, str(class_id)),
                        confidence=confidence,
                        bbox=[int(v) for v in box],
                        is_new=is_new,
                        raw_track_id=track.track_id,
                    )
                )

            # Drop labels for tracks JeztSort has deregistered, so the mapping
            # cannot grow without bound over a long run.
            #
            # Compared by MEMBERSHIP, not by size. The previous guard only ran
            # when len(labels) > len(tracker.tracks), so a deregistration that
            # coincided with a registration left the dead track's entry behind
            # for ever.
            live = set(tracker.tracks.keys())

            for track_id in [t for t in labels if t not in live]:
                del labels[track_id]

            for track_id in [t for t in coasting if t not in live]:
                del coasting[track_id]

            for track_id in [t for t in last_seen if t not in live]:
                del last_seen[track_id]

        return tracked

    # ------------------------------------------------------------------ reset
    def reset_camera(self, camera_id):
        """
        Drop all tracker and label state for one camera.

        Called when a camera stops being inferred - disabled in the dashboard,
        or all of its detection features switched off. Without this its tracks
        freeze mid-flight instead of ageing out, and a new person could inherit
        a stale ID when the camera is switched back on minutes later.
        """
        dropped = 0

        for group in (PERSON_GROUP, VEHICLE_GROUP, OBJECT_GROUP):

            key = (camera_id, group)

            if key in self._trackers:
                dropped += len(self._labels.get(key, {}))
                del self._trackers[key]
                self._labels.pop(key, None)

        return dropped

    # ------------------------------------------------------------------- tune
    def configure(self, *, max_disappeared=None, min_hits=None,
                  iou_thresh=None, dist_thresh=None, duplicate_iou_thresh=None,
                  vehicle_max_disappeared=None, vehicle_min_hits=None,
                  vehicle_iou_thresh=None, vehicle_dist_thresh=None):
        """
        Change tuning parameters for every tracker this instance owns, LIVE.

        JeztSort has no way to reconfigure an already-running ImprovedTracker,
        so this also drops every tracker and label this instance holds - every
        (camera, group) starts a fresh, empty tracker on its next update()
        call, built with the new parameters. Meant for interactive tuning
        (test_detect.py's sliders); production does not call this.

        Pass only the parameters you want to change; the rest keep their
        current value. Returns the number of track labels dropped.
        """
        if max_disappeared is not None:
            self._max_disappeared = max_disappeared

        if min_hits is not None:
            self._min_hits = min_hits

        if iou_thresh is not None:
            self._iou_thresh = iou_thresh

        if dist_thresh is not None:
            self._dist_thresh = dist_thresh

        if duplicate_iou_thresh is not None:
            self._duplicate_iou_thresh = duplicate_iou_thresh

        if vehicle_max_disappeared is not None:
            self._vehicle_max_disappeared = vehicle_max_disappeared

        if vehicle_min_hits is not None:
            self._vehicle_min_hits = vehicle_min_hits

        if vehicle_iou_thresh is not None:
            self._vehicle_iou_thresh = vehicle_iou_thresh

        if vehicle_dist_thresh is not None:
            self._vehicle_dist_thresh = vehicle_dist_thresh

        dropped = sum(len(labels) for labels in self._labels.values())

        self._trackers.clear()
        self._labels.clear()

        return dropped

    # ------------------------------------------------------------------- stats
    def active_track_count(self):
        return sum(len(labels) for labels in self._labels.values())
