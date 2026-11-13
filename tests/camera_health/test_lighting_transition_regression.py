"""The sharpness guard must not fire on a room going dark (ZAYED 2026-10-12).

    python -m pytest tests/camera_health/test_lighting_transition_regression.py

THE REGRESSION. The sharpness guard added on 2026-10-11 lets a collapsed image override
`structure_intact`, so that a partial lens cover is not excused by the edges still visible
around the hand. Its floor was set to 0.5, matching DEFOCUS_RATIO, on the assumption that
"re-lighting the SAME view barely moves sharpness".

Replaying the recorded clips through this module disproved that assumption. A room going dark
loses contrast with the light, and sharpness falls with it. On camera_01's 2026-10-09 19:50
lights-off the ratio reached 0.465-0.476 - under the old floor - so the guard overrode
structure_intact and started a tamper candidate on a light switch. Traced sample by sample
(tests/camera_health/run_tamper_replay.sh, clip B4_c01_1950):

    floor 0.50   t=13 struct_rejected bad=True  since=0.0   ... since reaches 10.0 s
    floor 0.35   t=13 structure_intact bad=None since=None  ... never starts

The guard was making the false positive it was meant to prevent MORE likely. The two
populations are cleanly separated, and the floor now sits in the gap:

    genuine cover  (camera_03 2026-09-30, raises)      0.088, 0.145, 0.271
    lights-off     (camera_01 2026-10-09, must not)    0.465, 0.476

Measured on the real clips, so these numbers are evidence, not fixtures.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from camera_health import detectors as D                                      # noqa: E402
from camera_health import state as state_mod                                  # noqa: E402

#: Replay of GENUINE_c03_0930_cover.mp4 - the frame-by-frame verified lens cover.
GENUINE_COVER_RATIOS = (0.088, 0.145, 0.271)
#: Replay of B4_c01_1950.mp4 - the classroom lights being switched off.
LIGHTS_OFF_RATIOS = (0.465, 0.476)


def _ready_state(monkeypatch, overlap, sharpness, baseline=4.0, distance=1.5):
    """A READY camera whose only varying inputs are the overlap and the sharpness ratio."""
    st = state_mod.CameraState("cam", 0.0)
    st.scene_phase = state_mod.PHASE_READY
    st.scene_baseline = np.ones(8, dtype=np.float32)
    st.edge_baseline = np.zeros((4, 4), dtype=bool)
    st.brightness_baseline = 100.0
    st.sharpness_baseline = baseline
    monkeypatch.setattr(D.metrics, "ensure_sample", lambda g: g)
    monkeypatch.setattr(D.metrics, "signature", lambda g: np.ones(8, dtype=np.float32))
    monkeypatch.setattr(D.metrics, "distance", lambda a, b: distance)
    monkeypatch.setattr(D.metrics, "sharpness", lambda g: sharpness * baseline)
    # Brightness moved, but not enough structure survived for the lighting excuse - the
    # condition every one of these samples was actually in.
    monkeypatch.setattr(D, "lighting_transition",
                        lambda s, g: (False, {"brightness_ratio": 1.4,
                                              "structural_overlap": overlap}))
    return st


# --------------------------------------------------------------- the boundary itself

def test_the_floor_separates_the_two_measured_populations():
    floor = D.LIGHTING_SHARPNESS_COLLAPSE_RATIO
    assert max(GENUINE_COVER_RATIOS) < floor, "a genuine cover must read as collapsed"
    assert min(LIGHTS_OFF_RATIOS) >= floor, "a room going dark must not"


def test_the_floor_keeps_a_margin_either_side():
    """Not merely between them: far enough from both that noise cannot cross it."""
    floor = D.LIGHTING_SHARPNESS_COLLAPSE_RATIO
    assert floor / max(GENUINE_COVER_RATIOS) >= 1.2, "too close to the covers"
    assert min(LIGHTS_OFF_RATIOS) / floor >= 1.2, "too close to the light switch"


# --------------------------------------------------------------- the gate's behaviour

@pytest.mark.parametrize("ratio", LIGHTS_OFF_RATIOS)
def test_lights_off_leaves_structure_intact_standing(monkeypatch, ratio):
    """THE REGRESSION. High overlap plus a dark room must stay protected: no candidate."""
    st = _ready_state(monkeypatch, overlap=0.78, sharpness=ratio)
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is None, detail
    assert detail.get("structure_intact") is True
    assert not detail.get("structure_intact_rejected")


@pytest.mark.parametrize("ratio", GENUINE_COVER_RATIOS)
def test_a_genuine_cover_still_overrides_structure_intact(monkeypatch, ratio):
    """The guard must keep doing its job: the 2026-09-30 cover still scores."""
    st = _ready_state(monkeypatch, overlap=0.90, sharpness=ratio)
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is True, detail
    assert detail.get("structure_intact_rejected") == "sharpness collapsed"


def test_lights_off_does_not_start_a_tamper_candidate(monkeypatch):
    """What the replay actually showed: under the old floor `since` began counting."""
    st = _ready_state(monkeypatch, overlap=0.78, sharpness=LIGHTS_OFF_RATIOS[0])
    for sample in range(12):
        bad, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), float(sample),
                                     obstructed=False)
        if bad is not None:
            st.tamper.update(bad, float(sample), D.TAMPER_SECONDS,
                             D.TAMPER_COOLDOWN_SECONDS, detail)
    assert st.tamper.since is None, "a light switch must never accumulate dwell"
    assert not st.tamper.active


def test_the_old_floor_is_what_broke_it(monkeypatch):
    """Proof the fixtures reproduce the defect rather than merely resembling it: with the
    floor back at 0.5, the lights-off sample is rejected and the candidate starts."""
    monkeypatch.setattr(D, "LIGHTING_SHARPNESS_COLLAPSE_RATIO", 0.5)
    st = _ready_state(monkeypatch, overlap=0.78, sharpness=LIGHTS_OFF_RATIOS[0])
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is True
    assert detail.get("structure_intact_rejected") == "sharpness collapsed"


# --------------------------------------------------------------- unchanged behaviour

def test_an_in_focus_scene_is_still_protected(monkeypatch):
    """The production false raises the structure gate exists for are untouched."""
    st = _ready_state(monkeypatch, overlap=0.90, sharpness=1.0)
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is None, detail
    assert detail.get("structure_intact") is True


def test_a_low_overlap_soft_sample_still_raises(monkeypatch):
    """Below the structure gate a cover-like (collapsed-sharpness) sample still raises."""
    st = _ready_state(monkeypatch, overlap=0.15, sharpness=0.10)
    verdict, _ = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is True


def test_a_low_overlap_in_focus_sample_needs_the_switch_off_to_raise(monkeypatch):
    """ZAYED 2026-11-13: in focus is rejected by TAMPER_REQUIRE_SOFT_IMAGE; switched off, the
    old rule applies unchanged."""
    st = _ready_state(monkeypatch, overlap=0.15, sharpness=1.0)
    verdict, _ = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is None
    monkeypatch.setattr(D, "TAMPER_REQUIRE_SOFT_IMAGE", False)
    st = _ready_state(monkeypatch, overlap=0.15, sharpness=1.0)
    verdict, _ = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is True


@pytest.mark.parametrize("ratio,raises", [
    # ZAYED 2026-11-13 measured on judged health_sample frames, sharpness / baseline:
    (0.04, True), (0.27, True),     # genuine 2026-09-30 camera_03 hand-on-lens cover
    (0.49, False), (1.00, False),   # weakest / typical false raise (people, lighting) 11-10..11-13
])
def test_measured_sharpness_ratios_split_covers_from_false_raises(monkeypatch, ratio, raises):
    st = _ready_state(monkeypatch, overlap=0.15, sharpness=ratio)
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert (verdict is True) is raises, detail


def test_no_sharpness_baseline_still_leaves_the_gate_as_it_was(monkeypatch):
    st = _ready_state(monkeypatch, overlap=0.90, sharpness=0.01, baseline=4.0)
    st.sharpness_baseline = None
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is None, detail
    assert detail.get("structure_intact") is True


def test_the_other_constants_are_untouched():
    assert D.DEFOCUS_RATIO == 0.50
    assert D.TAMPER_DISTANCE == 0.65
    if "CAMERA_HEALTH_TAMPER_STRUCTURE_INTACT_MIN" not in os.environ:
        assert D.TAMPER_STRUCTURE_INTACT_MIN == 0.65
    if "CAMERA_HEALTH_STRUCTURAL_OVERLAP_LIGHTING_MIN" not in os.environ:
        assert D.STRUCTURAL_OVERLAP_LIGHTING_MIN == 0.80
