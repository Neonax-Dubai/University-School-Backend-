"""Camera Tampering requires the baseline structure to be GONE (ZAYED, 2026-10-04).

structural_overlap is the share of the baseline's strong edges still present. This module's
own calibration puts a genuine lens cover at 0.10-0.17 and lights on/off at 0.86-0.87, so an
overlap at or above STRUCTURAL_OVERLAP_LIGHTING_MIN means the view was never replaced.

The lighting guard alone could not refuse these: it also demands a brightness move of
LIGHTING_MIN_BRIGHTNESS_RATIO, and the false raises observed in production had brightness
essentially unchanged (1.00-1.23) with 0.80-0.93 of the structure intact.

Runs without the lighting fixture images, which are not in this checkout.
"""
import numpy as np
import pytest

from camera_health import detectors, state as state_mod


def _ready_state(monkeypatch, overlap, distance=0.9):
    st = state_mod.CameraState("cam", 0.0)
    st.scene_phase = state_mod.PHASE_READY
    st.scene_baseline = np.ones(8, dtype=np.float32)
    st.edge_baseline = np.zeros((4, 4), dtype=bool)
    st.brightness_baseline = 100.0

    monkeypatch.setattr(detectors.metrics, "ensure_sample", lambda g: g)
    monkeypatch.setattr(detectors.metrics, "signature", lambda g: np.ones(8, dtype=np.float32))
    monkeypatch.setattr(detectors.metrics, "distance", lambda a, b: distance)
    # The guard refuses (brightness did not move); only the overlap varies.
    monkeypatch.setattr(detectors, "lighting_transition",
                        lambda s, g: (False, {"brightness_ratio": 1.00,
                                              "structural_overlap": overlap}))
    return st


def test_structure_still_present_is_not_tampering(monkeypatch):
    """0.90 of the baseline edges remain: nothing covered the lens.

    The answer is None, not False: False would clear an accumulating candidate (see
    test_gate_does_not_starve_a_real_cover).
    """
    st = _ready_state(monkeypatch, overlap=0.90)
    bad, detail = detectors.tamper_state(st, np.zeros((4, 4), np.uint8), 1.0, obstructed=False)
    assert bad is None
    assert detail["structure_intact"] is True
    assert st.structure_intact_rejects == 1


def test_cover_range_overlap_still_raises(monkeypatch):
    """0.15 overlap is the measured signature of a real cover - must still raise."""
    st = _ready_state(monkeypatch, overlap=0.15)
    bad, detail = detectors.tamper_state(st, np.zeros((4, 4), np.uint8), 1.0, obstructed=False)
    assert bad is True
    assert "structure_intact" not in detail
    assert st.structure_intact_rejects == 0


def test_gate_sits_exactly_on_the_constant(monkeypatch):
    """Just under the constant still raises; at it does not."""
    edge = detectors.TAMPER_STRUCTURE_INTACT_MIN
    below = _ready_state(monkeypatch, overlap=edge - 0.001)
    assert detectors.tamper_state(below, np.zeros((4, 4), np.uint8), 1.0, False)[0] is True
    at = _ready_state(monkeypatch, overlap=edge)
    assert detectors.tamper_state(at, np.zeros((4, 4), np.uint8), 1.0, False)[0] is None


def test_production_false_raises_are_suppressed(monkeypatch):
    """The four 10-03 camera_03 raises, replayed from their stored measurements."""
    for distance, overlap in ((0.6527, 0.925), (0.6533, 0.924), (0.6571, 0.910),
                              (0.6510, 0.901), (0.8131, 0.842), (1.1147, 0.813),
                              (0.9624, 0.811), (0.7117, 0.799), (0.9148, 0.738),
                              (0.6529, 0.735), (0.7430, 0.680), (0.6537, 0.665)):
        st = _ready_state(monkeypatch, overlap=overlap, distance=distance)
        bad, _ = detectors.tamper_state(st, np.zeros((4, 4), np.uint8), 1.0, False)
        assert bad is None, (distance, overlap)


def test_production_genuine_raises_are_kept(monkeypatch):
    """The most cover-like stored raises must survive the gate."""
    for distance, overlap in ((1.4637, 0.193), (1.0628, 0.303), (1.2936, 0.315)):
        st = _ready_state(monkeypatch, overlap=overlap, distance=distance)
        bad, _ = detectors.tamper_state(st, np.zeros((4, 4), np.uint8), 1.0, False)
        assert bad is True, (distance, overlap)


def test_gate_does_not_starve_a_real_cover(monkeypatch):
    """The measured 2026-09-30 16:17 cover on camera_01 must still confirm.

    Samples every 2 s, overlap taken from the footage. Two frames mid-cover still overlap
    0.94 (a hand entering the view before it blocks it). Returning False there would reset
    Condition.since and the raise would hang on zero margin; None keeps the candidate.
    """
    from camera_health import state as sm

    trace = [(0, 1.1643, 0.227), (2, 1.3558, 0.597), (4, 0.9611, 0.943),
             (6, 0.9492, 0.943), (8, 1.3093, 0.657), (10, 1.1751, 0.105),
             (12, 1.3651, 0.231), (14, 1.2404, 0.157), (16, 1.1064, 0.565)]
    st = sm.CameraState("cam", 0.0)
    st.scene_phase = sm.PHASE_READY
    st.scene_baseline = np.ones(8, dtype=np.float32)
    st.edge_baseline = np.zeros((4, 4), dtype=bool)
    st.brightness_baseline = 100.0
    monkeypatch.setattr(detectors.metrics, "ensure_sample", lambda g: g)
    monkeypatch.setattr(detectors.metrics, "signature", lambda g: np.ones(8, dtype=np.float32))

    actions = []
    for offset, distance, overlap in trace:
        monkeypatch.setattr(detectors.metrics, "distance", lambda a, b, d=distance: d)
        monkeypatch.setattr(detectors, "lighting_transition",
                            lambda s, g, o=overlap: (False, {"brightness_ratio": 1.03,
                                                             "structural_overlap": o}))
        bad, detail = detectors.tamper_state(st, np.zeros((4, 4), np.uint8), float(offset), False)
        if bad is not None:
            actions.append((offset, st.tamper.update(bad, float(offset),
                                                     detectors.TAMPER_SECONDS,
                                                     detectors.TAMPER_COOLDOWN_SECONDS, detail)))
    raises = [o for o, a in actions if a == "raise"]
    assert raises, f"the real cover must still raise; actions={actions}"
    # The cover ran t=0-16 s. It confirms at t=10: the candidate opens on the first replaced
    # sample (t=0, overlap 0.227) and survives the two intact frames at t=4/6 because those
    # answer None rather than False. Returning False there would restart the clock at t=8 and
    # the incident would confirm at t=16 - the very last sample - or not at all if the hand
    # had come away a second sooner.
    assert raises[0] <= 12, f"must confirm well inside the 16 s cover, got {raises[0]}"


def test_a_candidate_does_not_outlive_the_persistence_window(monkeypatch):
    """A stale candidate must not let a single later sample raise without its own 8 s."""
    st = _ready_state(monkeypatch, overlap=0.50, distance=0.9)
    detectors.tamper_state(st, np.zeros((4, 4), np.uint8), 0.0, False)
    st.tamper.update(True, 0.0, detectors.TAMPER_SECONDS,
                     detectors.TAMPER_COOLDOWN_SECONDS, {})
    assert st.tamper.since == 0.0

    # Structure intact for longer than the persistence window -> candidate dropped.
    monkeypatch.setattr(detectors, "lighting_transition",
                        lambda s, g: (False, {"brightness_ratio": 1.0,
                                              "structural_overlap": 0.95}))
    for t in (1.0, 5.0, 20.0):
        detectors.tamper_state(st, np.zeros((4, 4), np.uint8), t, False)
    assert st.tamper.since is None
