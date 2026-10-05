"""
Camera-health detectors: thresholds, windows and the per-sample decisions.

Each *_state() function looks at one sample (and the camera's own baselines)
and answers "is this detector's fault present in THIS sample?" - True, False,
or None for "no decision possible". Persistence, cooldown and the one-raise /
one-recovery rule belong to state.Condition, never to a detector.

Thresholds are the approved Dubai values (asserted by the acceptance tests):
OBSTRUCTION_EDGE_MAX 0.15, DEFOCUS_RATIO 0.50, TAMPER_DISTANCE 0.65, tamper
persistence 8 s (validated on real CAM-R25 replays: 5 s fired on people walking,
8 s did not, genuine covers still did).

Two deliberately differ from that baseline at Zayed (2026-10-05):
SIGNAL_LOSS_SECONDS 120.0 (was 5.0) and the new TAMPER_STRUCTURE_INTACT_MIN.
Both are below, with the measurements that set them. The signal-loss tests derive
their timings from the constant rather than pinning 5 s, so the contract they
assert - a gap longer than the threshold is an outage - holds at either value.
"""

import os

import numpy as np

from . import metrics
from .state import PHASE_LEARNING, PHASE_READY, PHASE_STARTING


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------ signal loss
# ZAYED 2026-10-05. Was a hard-coded 5.0 with no way to tune it. Every restart of the
# inference service, every relay blip and every replay run dropped all three streams for a
# few seconds, and each one became a CRITICAL "Camera Signal Lost" alarm on all three
# cameras at once - 83 of them in the stored history. Here an operator cares that a camera
# is really gone, not that a frame was late, so the default is two minutes and the env var
# still allows a site to choose otherwise.
SIGNAL_LOSS_SECONDS = _env_float("CAMERA_HEALTH_SIGNAL_LOSS_SECONDS", 120.0)
SIGNAL_LOSS_COOLDOWN_SECONDS = _env_float("CAMERA_HEALTH_SIGNAL_LOSS_COOLDOWN_SECONDS", 300.0)

# ------------------------------------------------------------ obstruction
#: Obstructed when at most this share of the frame still carries structure.
OBSTRUCTION_EDGE_MAX = 0.15
OBSTRUCTION_SECONDS = 5.0
OBSTRUCTION_COOLDOWN_SECONDS = 0.0

# ------------------------------------------------------------ defocus
DEFOCUS_RATIO = 0.50
DEFOCUS_SECONDS = 5.0
DEFOCUS_COOLDOWN_SECONDS = 300.0
#: Structured samples used to learn the sharpness baseline (median).
BASELINE_SAMPLES = 10
SHARPNESS_ADAPT_ALPHA = 0.02

# ------------------------------------------------------------ scene change
TAMPER_DISTANCE = 0.65
TAMPER_SECONDS = _env_float("CAMERA_HEALTH_TAMPER_SECONDS", 8.0)
TAMPER_COOLDOWN_SECONDS = _env_float("CAMERA_HEALTH_TAMPER_COOLDOWN_SECONDS", 0.0)
#: Samples discarded at start-up and after every baseline reset (the live
#: defect of 2026-09-15 took the baseline from the first decoded sample).
SCENE_WARMUP_SAMPLES = 3
#: Mutually stable samples averaged into the scene baseline.
SCENE_BASELINE_SAMPLES = 5
SCENE_STABLE_DISTANCE = 0.25
#: Healthy samples this close to the baseline nudge it, so slow daylight drift
#: is absorbed. A candidate, an incident or a people-sized change never does.
SCENE_ADAPT_DISTANCE = 0.15
SCENE_ADAPT_ALPHA = 0.02
#: Contract constant. Until 2026-09-19 an incident active this long was forced
#: through a re-baseline, which announced a recovery for a view that never
#: recovered. An active incident is now never re-baselined; this value is kept
#: only because the lifecycle tests size their scenarios from it.
SCENE_REBASELINE_AFTER_SECONDS = 900.0

# ------------------------------------------------------------ lighting guard
STRUCTURAL_OVERLAP_LIGHTING_MIN = _env_float("CAMERA_HEALTH_STRUCTURAL_OVERLAP_LIGHTING_MIN", 0.80)

# ZAYED 2026-10-05. Camera Tampering means the view was REPLACED, so a sample whose structure
# still overlaps the baseline this much is not interference whatever moved the signature.
# Calibrated on this site's own footage, not inherited: the real lens cover of 2026-09-30
# 16:16:56-16:17:12 on camera_01 measured overlap 0.105-0.597 across six of its nine samples,
# while every false raise in the stored history sat at 0.665-0.925. 0.65 separates them and
# still confirms that cover (see tests/camera_health/test_camera_tamper_structure_gate.py).
TAMPER_STRUCTURE_INTACT_MIN = _env_float("CAMERA_HEALTH_TAMPER_STRUCTURE_INTACT_MIN", 0.65)
LIGHTING_MIN_BRIGHTNESS_RATIO = _env_float("CAMERA_HEALTH_LIGHTING_MIN_BRIGHTNESS_RATIO", 1.3)


def scene_signature(gray):
    return metrics.signature(metrics.ensure_sample(gray))


def scene_distance(a, b):
    return metrics.distance(a, b)


# ============================================================ obstruction
def obstruction_state(share):
    return share <= OBSTRUCTION_EDGE_MAX, {"edge_density": round(share, 3),
                                           "obstruction_pct": round(100.0 * (1.0 - share), 1)}


# ============================================================ defocus
def defocus_state(state, value, obstructed):
    """Decide defocus from this sample's sharpness; learn the baseline first."""
    if obstructed:
        return None, {"skipped": "unstructured sample"}
    if state.sharpness_baseline is None:
        state.sharpness_samples.append(value)
        if len(state.sharpness_samples) >= BASELINE_SAMPLES:
            state.sharpness_baseline = float(np.median(state.sharpness_samples))
            state.sharpness_samples = []
        return None, {"learning": len(state.sharpness_samples)}
    baseline = state.sharpness_baseline
    ratio = value / baseline if baseline > 1e-9 else 1.0
    bad = ratio < DEFOCUS_RATIO
    if (not bad and not state.defocus.active and state.defocus.since is None
            and 0.8 <= ratio <= 1.25):
        state.sharpness_baseline = (1.0 - SHARPNESS_ADAPT_ALPHA) * baseline + SHARPNESS_ADAPT_ALPHA * value
    return bad, {"ratio": round(ratio, 3), "sharpness": round(value, 5),
                 "baseline": round(baseline, 5), "threshold": DEFOCUS_RATIO}


# ============================================================ lighting guard
def lighting_transition(state, gray):
    """Is the difference from the baseline a lighting change of the SAME view?

    Both must hold: global brightness moved by at least
    LIGHTING_MIN_BRIGHTNESS_RATIO, and at least STRUCTURAL_OVERLAP_LIGHTING_MIN
    of the dimmer view's strongest edges lie on the other view's edges.
    Measured on the real fixtures: lights on/off 1.60-1.65 / 0.86-0.87; covers
    1.03-1.16 / 0.10-0.17; people walking 1.05 / 0.17.
    """
    if state.edge_baseline is None or state.brightness_baseline is None:
        return False, {"lighting_check": "no baseline"}
    gray = metrics.ensure_sample(gray)
    now_brightness = metrics.brightness(gray)
    base_brightness = state.brightness_baseline
    ratio = max(now_brightness, base_brightness) / max(min(now_brightness, base_brightness), 1.0)
    overlap = metrics.structural_overlap(metrics.edge_map(gray), now_brightness,
                                         state.edge_baseline, base_brightness)
    ok = ratio >= LIGHTING_MIN_BRIGHTNESS_RATIO and overlap >= STRUCTURAL_OVERLAP_LIGHTING_MIN
    return ok, {"brightness_ratio": round(ratio, 3), "structural_overlap": round(overlap, 3)}


def _guard_blocked(state, obstructed):
    """The guard may only explain a change nothing else already explains."""
    return (obstructed or state.tamper.active or state.obstruction.active
            or state.obstruction.since is not None or state.signal.active)


# ============================================================ scene change
def _learn_scene(state, gray, sig):
    if state.scene_warmup < SCENE_WARMUP_SAMPLES:
        state.scene_warmup += 1
        state.scene_phase = PHASE_STARTING if state.scene_warmup < SCENE_WARMUP_SAMPLES else PHASE_LEARNING
        return
    state.scene_phase = PHASE_LEARNING
    if state.scene_candidates:
        mean = np.mean(state.scene_candidates, axis=0)
        mean -= mean.mean()
        norm = float(np.linalg.norm(mean))
        if norm < 1e-6 or metrics.distance(sig, mean / norm) > SCENE_STABLE_DISTANCE:
            state.scene_candidates, state.scene_candidate_grays = [], []   # not stable yet
    state.scene_candidates.append(sig)
    state.scene_candidate_grays.append(gray.astype(np.float32))
    if len(state.scene_candidates) < SCENE_BASELINE_SAMPLES:
        return
    baseline = np.mean(state.scene_candidates, axis=0)
    baseline -= baseline.mean()
    baseline /= max(float(np.linalg.norm(baseline)), 1e-6)
    mean_gray = np.clip(np.mean(state.scene_candidate_grays, axis=0), 0, 255).astype(np.uint8)
    state.scene_baseline = baseline.astype(np.float32)
    state.edge_baseline = metrics.edge_map(mean_gray)
    state.brightness_baseline = metrics.brightness(mean_gray)
    state.scene_phase = PHASE_READY
    state.scene_candidates, state.scene_candidate_grays = [], []


def tamper_state(state, gray, now, obstructed):
    """Scene-change decision for one sample: (bad | None, detail).

    No decision while the sample is featureless (obstruction owns that) or the
    baseline is still being learned. An apparent change that the lighting
    guard explains re-learns the baseline on the new light instead of raising;
    an ACTIVE incident is never re-baselined, because that would fake a
    recovery.
    """
    gray = metrics.ensure_sample(gray)
    if obstructed:
        if state.scene_phase != PHASE_READY:
            state.scene_rejected += 1
        return None, {"skipped": "unstructured sample"}
    sig = metrics.signature(gray)
    if sig is None:
        if state.scene_phase != PHASE_READY:
            state.scene_rejected += 1
        return None, {"skipped": "featureless sample"}
    if state.scene_phase != PHASE_READY:
        _learn_scene(state, gray, sig)
        return None, {"phase": state.scene_phase}

    d = metrics.distance(sig, state.scene_baseline)
    state.last_distance = d
    detail = {"distance": round(d, 4), "threshold": TAMPER_DISTANCE}
    bad = d >= TAMPER_DISTANCE

    if bad and not _guard_blocked(state, obstructed):
        ok, check = lighting_transition(state, gray)
        if ok:
            state.illumination_resets += 1
            state.reset_image_baselines()
            detail.update(check)
            detail["illumination_transition"] = True
            return False, detail
        detail["lighting_check"] = check

        # ZAYED. Camera Tampering means the view was REPLACED. structural_overlap is the
        # share of the baseline's strong edges still present, and this module's own
        # calibration puts a genuine cover at 0.10-0.17 (lights on/off 0.86-0.87). So an
        # overlap at or above the lighting minimum says the baseline structure is still
        # there: whatever moved the signature past TAMPER_DISTANCE, nothing covered the lens.
        #
        # The lighting guard above could not excuse these because it also demands a
        # brightness move of LIGHTING_MIN_BRIGHTNESS_RATIO, and the observed false raises
        # had brightness essentially unchanged (ratio 1.00-1.23) with structure 0.80-0.93
        # intact - a scene that drifted, a replayed clip, a baseline learned on a transient
        # frame at stream start. None of them is interference, and each one cost an operator
        # a critical alarm.
        overlap = check.get("structural_overlap")
        if overlap is not None and overlap >= TAMPER_STRUCTURE_INTACT_MIN:
            state.structure_intact_rejects += 1
            detail["structure_intact"] = True
            if state.structure_intact_since is None:
                state.structure_intact_since = now
            # NONE, not False. Condition.update(bad=False) clears `since`, and a real cover
            # contains transitional frames whose structure still overlaps (measured on the
            # 2026-09-30 16:17 cover: overlap 0.94 at two samples inside a 0.10-0.23 episode).
            # Returning False there resets the 8 s candidate mid-cover and left the genuine
            # raise on zero margin. The manager skips the condition entirely on None, so the
            # candidate survives the gap and the cover still confirms - two seconds sooner.
            #
            # A candidate that has outlived the persistence window while the structure stayed
            # intact is protecting nothing, so it is dropped: otherwise a later unrelated
            # sample could inherit its age and raise without serving its own 8 s.
            if (not state.tamper.active and state.tamper.since is not None
                    and now - state.structure_intact_since >= TAMPER_SECONDS):
                state.tamper.since = None
            return None, detail
        state.structure_intact_since = None
    elif (not bad and not state.tamper.active and state.tamper.since is None
          and d < SCENE_ADAPT_DISTANCE):
        adapted = (1.0 - SCENE_ADAPT_ALPHA) * state.scene_baseline + SCENE_ADAPT_ALPHA * sig
        adapted -= adapted.mean()
        state.scene_baseline = (adapted / max(float(np.linalg.norm(adapted)), 1e-6)).astype(np.float32)
    return bad, detail
