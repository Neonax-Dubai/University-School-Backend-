"""
Camera-health state: one Condition per detector, one CameraState per camera.

Recreated for Zayed University (2026-09-22) from the proven Dubai contract -
the production callers in multicam_inf.py, the seven camera-health test files
and the 2026-09-16 / 2026-09-19 design notes. The original package lived only
on the Dubai production server and was never committed.

A Condition turns a per-sample yes/no into at most ONE raise and ONE recovery
per incident:

  * a fault must persist `persistence` seconds, measured from its FIRST bad
    sample, before it is announced;
  * a recovery waits the SAME window, measured from the first GOOD sample
    (the 2026-09-16 defect announced the recovery one second after the raise);
  * a fault returning during that hold cancels the recovery - a flickering
    condition is one incident, not a stream of pairs;
  * `cooldown` suppresses a new raise that follows a recovery too closely.
"""

import collections

PHASE_STARTING = "starting"      # warm-up samples, discarded
PHASE_LEARNING = "learning"      # collecting mutually stable baseline samples
PHASE_READY = "ready"            # baseline valid; decisions are made

#: Availability: share of recent one-second slots in which frames arrived.
AVAILABILITY_WINDOW_SECONDS = 300
AVAILABILITY_MIN_SLOTS = 60

#: Delivered frame rate from the reader's own counter.
FPS_WINDOW_SECONDS = 10.0
FPS_MIN_SPAN_SECONDS = 2.0
#: No counter observation for this long means the rate is unknown, not zero.
FPS_STALE_SECONDS = 5.0

#: An image measurement older than this is no longer a current reading.
IMAGE_STALE_SECONDS = 15.0


class Condition:
    """One detector's persisted state for one camera."""

    def __init__(self, name):
        self.name = name
        self.active = False
        self.since = None          # first bad sample of the current candidate
        self.hold_since = None     # first good sample while active
        self.raised_at = None
        self.recovered_at = None
        self.raises = 0
        self.recoveries = 0
        self.detail = {}

    def update(self, bad, at, persistence, cooldown, detail, recover_persistence=None):
        """Advance with one sample. Returns "raise", "recover" or None."""
        self.detail = dict(detail or {})
        if bad:
            if self.active:
                self.hold_since = None          # fault back during the hold
                return None
            if self.since is None:
                self.since = at
            if at - self.since < persistence:
                return None
            if cooldown and self.recovered_at is not None and at - self.recovered_at < cooldown:
                return None
            self.active = True
            self.raised_at = at
            self.hold_since = None
            self.raises += 1
            return "raise"

        self.since = None
        if not self.active:
            return None
        hold = persistence if recover_persistence is None else recover_persistence
        if self.hold_since is None:
            self.hold_since = at
        if at - self.hold_since < hold:
            return None
        self.active = False
        self.hold_since = None
        self.recovered_at = at
        self.recoveries += 1
        return "recover"

    def as_dict(self):
        return {"active": self.active, "raises": self.raises,
                "recoveries": self.recoveries, "detail": dict(self.detail)}


class CameraState:
    """Everything camera health knows about one camera. Mutated under the manager's lock."""

    def __init__(self, camera_id, created_at):
        self.camera_id = camera_id
        self.created_at = created_at

        self.signal = Condition("camera_signal_loss")
        self.obstruction = Condition("camera_obstruction")
        self.defocus = Condition("camera_defocus")
        self.tamper = Condition("camera_tamper")

        # ---- stream counters (note_streams) ---------------------------------
        self.last_frames = None
        self.last_progress_at = None
        self.last_note_at = None
        self.slots = collections.OrderedDict()      # int(second) -> delivered?
        self.fps_window = collections.deque()       # (t, frames_read)

        # ---- image measurements (the 1 Hz sample) --------------------------
        self.frame_size = None
        self.profile_reset_pending = None
        self.profile_resets = 0
        self.last_sample_at = None
        self.last_structured_share = None
        self.last_sharpness = None
        self.last_brightness = None
        self.last_distance = None

        # ---- baselines -------------------------------------------------------
        self.sharpness_baseline = None
        self.sharpness_samples = []
        self.scene_baseline = None
        self.edge_baseline = None
        self.brightness_baseline = None
        self.scene_phase = PHASE_STARTING
        self.scene_warmup = 0
        self.scene_candidates = []
        self.scene_candidate_grays = []
        self.scene_rejected = 0
        self.scene_rebuilds = 0
        self.illumination_resets = 0

    def scene_baseline_valid(self):
        return self.scene_baseline is not None and self.scene_phase == PHASE_READY

    def reset_image_baselines(self):
        """Forget every image baseline and re-learn from the next sample."""
        self.sharpness_baseline = None
        self.sharpness_samples = []
        self.reset_scene_baseline()

    def reset_scene_baseline(self):
        self.scene_baseline = None
        self.edge_baseline = None
        self.brightness_baseline = None
        self.scene_phase = PHASE_STARTING
        self.scene_warmup = 0
        self.scene_candidates = []
        self.scene_candidate_grays = []

    def observed(self):
        return self.last_note_at is not None or self.last_sample_at is not None
