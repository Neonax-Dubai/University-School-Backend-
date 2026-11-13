"""The structure-intact gate, and why a collapsed image must not satisfy it (ZAYED 2026-10-11).

THE REGRESSION. `structure_intact` exists for a good reason: an overlap at or above
TAMPER_STRUCTURE_INTACT_MIN is supposed to mean "the baseline's edges are still there, so
nothing covered the lens". That holds for the production false raises it was built for -
a drifted scene, a replayed clip, a baseline learned on a transient frame - which had
brightness 1.00-1.23 and 0.80-0.93 of the structure intact, all of them in focus.

It does NOT hold for a PARTIAL cover. Traced sample by sample on the real 2026-09-30
camera_03 episode, the genuinely occluded phase lasts 6 s (t=21-26, overlap 0.10-0.62).
From t=27 the hand slides partly off, overlap creeps back over 0.65, every later sample
returns None, and no further bad sample ever arrives to carry the candidate past
TAMPER_SECONDS. The cover is two seconds short of raising and then the lighting branch
drops the candidate entirely:

    t     distance  overlap  sharpness_ratio   branch                    since
    20      0.842    0.856       0.604         structure_intact -> None    -
    21      1.260    0.163         -           scored True                0.0
    ...
    26      1.534    0.617         -           scored True                5.0
    27      1.539    0.651       0.085         structure_intact -> None   6.0
    28      1.557    0.840       0.141         structure_intact -> None   7.0
    29      1.456    0.848       0.200         structure_intact -> None   8.0
    30      1.515    0.859       0.180         structure_intact -> None   9.0
    31      0.915      -           -           LIGHTING -> rebaseline   DROPPED

Sharpness separates the two populations cleanly, and it is the same metric and the same
floor the lighting guard already uses:

    wrongly rejected (real cover) : 0.085, 0.141, 0.200
    rightly rejected              : 0.604 (c03 t=20 transitional), 1.321 (c01 t=36)

So the gate keeps its meaning - "the structure is still there" - but is no longer allowed to
conclude "nothing covered the lens" from an image that has gone soft.
"""
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from camera_health import detectors as D                                      # noqa: E402
from camera_health import metrics as M                                        # noqa: E402
from camera_health import state as state_mod                                  # noqa: E402

FIX = os.path.join(HERE, "test_fixtures", "camera_tamper_sharpness")


def fixture(name):
    img = cv2.imread(os.path.join(FIX, name + ".png"), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (180, 320), name
    return img


def _sharpness_ratio(sample, baseline_img):
    st = state_mod.CameraState("cam", 0.0)
    for _ in range(D.BASELINE_SAMPLES):
        D.defocus_state(st, M.sharpness(baseline_img), obstructed=False)
    return M.sharpness(sample) / st.sharpness_baseline


# --------------------------------------------------------------- measured data

@pytest.mark.parametrize("name,expect_collapsed", [
    ("c03_cover_looks_like_lighting", True),    # t=28, real cover, wrongly rejected
    ("c03_t28_cover_structure_intact", True),   # t=28 sample, structure_intact branch
    ("c03_t20_transitional", False),            # t=20, before the occlusion
    ("c01_t36_structure_intact", False),        # camera_01, in focus
])
def test_the_two_populations_sit_either_side_of_the_floor(name, expect_collapsed):
    ratio = _sharpness_ratio(fixture(name), fixture("c03_clean"))
    collapsed = ratio < D.LIGHTING_SHARPNESS_COLLAPSE_RATIO
    assert collapsed is expect_collapsed, f"{name}: sharpness ratio {ratio:.3f}"


# ------------------------------------------------------- the gate's own logic

def _gate_state(monkeypatch, overlap, distance=0.9, sharpness_baseline=None):
    """A READY state whose only varying inputs are overlap and the sharpness baseline."""
    st = state_mod.CameraState("cam", 0.0)
    st.scene_phase = state_mod.PHASE_READY
    st.scene_baseline = np.ones(8, dtype=np.float32)
    st.edge_baseline = np.zeros((4, 4), dtype=bool)
    st.brightness_baseline = 100.0
    st.sharpness_baseline = sharpness_baseline
    monkeypatch.setattr(D.metrics, "ensure_sample", lambda g: g)
    monkeypatch.setattr(D.metrics, "signature", lambda g: np.ones(8, dtype=np.float32))
    monkeypatch.setattr(D.metrics, "distance", lambda a, b: distance)
    monkeypatch.setattr(D, "lighting_transition",
                        lambda s, g: (False, {"brightness_ratio": 1.00,
                                              "structural_overlap": overlap}))
    return st


def test_an_in_focus_scene_with_structure_intact_is_still_not_tampering(monkeypatch):
    """THE NEGATIVE CASE. This is the production false raise the gate exists for."""
    st = _gate_state(monkeypatch, overlap=0.90, sharpness_baseline=4.0)
    monkeypatch.setattr(D.metrics, "sharpness", lambda g: 4.0)        # ratio 1.0, in focus
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is None, detail
    assert detail.get("structure_intact") is True


def test_a_collapsed_image_may_not_claim_structure_intact(monkeypatch):
    """THE POSITIVE CASE. Same overlap, but the image has gone soft."""
    st = _gate_state(monkeypatch, overlap=0.90, sharpness_baseline=4.0)
    monkeypatch.setattr(D.metrics, "sharpness", lambda g: 0.4)        # ratio 0.1, collapsed
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is True, detail
    assert not detail.get("structure_intact")


def test_no_sharpness_baseline_leaves_the_gate_exactly_as_it_was(monkeypatch):
    """Missing data must never be what switches a safeguard off - and the existing
    structure-gate suite relies on this, because it never sets a baseline."""
    st = _gate_state(monkeypatch, overlap=0.90, sharpness_baseline=None)
    monkeypatch.setattr(D.metrics, "sharpness", lambda g: 0.0)
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is None, detail
    assert detail.get("structure_intact") is True


def test_a_degenerate_baseline_leaves_the_gate_as_it_was(monkeypatch):
    st = _gate_state(monkeypatch, overlap=0.90, sharpness_baseline=0.0)
    monkeypatch.setattr(D.metrics, "sharpness", lambda g: 0.0)
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is None, detail


def test_overlap_below_the_gate_raises_when_the_image_went_soft(monkeypatch):
    """Below the structure gate a changed view that ALSO lost sharpness is a cover."""
    st = _gate_state(monkeypatch, overlap=0.15, sharpness_baseline=4.0)
    monkeypatch.setattr(D.metrics, "sharpness", lambda g: 0.4)        # ratio 0.10, soft
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is True, detail


def test_overlap_below_the_gate_in_focus_is_rejected(monkeypatch):
    """ZAYED 2026-11-13 (TAMPER_REQUIRE_SOFT_IMAGE): low overlap with a sharp image is people
    near the lens or a lighting change, not a cover."""
    st = _gate_state(monkeypatch, overlap=0.15, sharpness_baseline=4.0)
    monkeypatch.setattr(D.metrics, "sharpness", lambda g: 4.0)        # ratio 1.0, in focus
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is None, detail
    assert "in_focus_rejected" in detail, detail


def test_overlap_below_the_gate_in_focus_raises_with_the_switch_off(monkeypatch):
    """CAMERA_HEALTH_TAMPER_REQUIRE_SOFT_IMAGE=0 restores the previous rule exactly."""
    monkeypatch.setattr(D, "TAMPER_REQUIRE_SOFT_IMAGE", False)
    st = _gate_state(monkeypatch, overlap=0.15, sharpness_baseline=4.0)
    monkeypatch.setattr(D.metrics, "sharpness", lambda g: 4.0)
    verdict, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert verdict is True, detail


def test_the_rejection_is_recorded_for_an_operator(monkeypatch):
    st = _gate_state(monkeypatch, overlap=0.90, sharpness_baseline=4.0)
    monkeypatch.setattr(D.metrics, "sharpness", lambda g: 0.4)
    _, detail = D.tamper_state(st, np.zeros((4, 4), np.uint8), 10.0, obstructed=False)
    assert detail.get("structure_intact_rejected") == "sharpness collapsed", detail


def test_existing_constants_are_untouched():
    if "CAMERA_HEALTH_TAMPER_STRUCTURE_INTACT_MIN" not in os.environ:
        assert D.TAMPER_STRUCTURE_INTACT_MIN == 0.65
    if "CAMERA_HEALTH_TAMPER_SECONDS" not in os.environ:
        assert D.TAMPER_SECONDS == 8.0
    assert D.TAMPER_DISTANCE == 0.65
    assert D.DEFOCUS_RATIO == 0.50
