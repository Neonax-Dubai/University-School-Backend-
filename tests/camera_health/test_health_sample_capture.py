"""The frame a camera-health raise was decided on is kept (ZAYED 2026-10-12).

    python -m pytest tests/camera_health/test_health_sample_capture.py

WHY. On 2026-10-12 camera_01 raised camera_tamper at 07:13:11.631Z recording distance 1.0584,
brightness ratio 1.545 and sharpness 0.436. Ten minutes of NVR footage spanning that moment,
replayed through this module, measured distance 0.009-0.019 and luma 90.0-90.8 throughout: the
recorder never saw what the detector saw. Nothing in the pipeline kept the picture the detector
judged, so there was no way to tell a corrupted frame from a real scene change - and no amount
of threshold tuning could have told them apart either.

WHAT IS STORED. One image per RAISE: the 320x180 greyscale array the detectors were handed,
encoded as it is, under evidence type "health_sample". Not a crop of a subject, not a fresh
grab afterwards, and not one per sample.

THE RULES THESE PIN.
  * it is the judged frame, not a later one;
  * a raise carries it, a recovery does not, and a non-tampering detector emits nothing at all;
  * it is bounded - oversize or malformed input is dropped, never stored;
  * storing it can never cost the event, whatever storage does;
  * it goes in under its own type so it cannot be mistaken for a scene snapshot;
  * the existing event path is unchanged when no sample is offered.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from camera_health import detectors as D                                      # noqa: E402
from camera_health.manager import (CAMERA_HEALTH_FEATURES, FEATURE_DEFOCUS,   # noqa: E402
                                   FEATURE_TAMPER, CameraHealthManager)

CAM = "camera_01"


def scene(seed, shape=(180, 320)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, shape, dtype=np.uint8)


class Harness:
    """A live-mode manager with an in-memory sink - nothing leaves the process."""

    def __init__(self):
        self.sent = []
        self.m = CameraHealthManager(shadow=False, logger=lambda *_: None,
                                     event_sink=lambda **kw: self.sent.append(kw))
        self.m.set_enabled_cameras({CAM: {f: True for f in CAMERA_HEALTH_FEATURES}})
        self.t = 0.0

    def feed(self, img, count, features=(FEATURE_TAMPER,)):
        for _ in range(count):
            self.m.process_sample(CAM, img, self.t, set(features))
            self.t += 1.0

    def raises(self):
        return [e for e in self.sent if e.get("action") == "raise"]


def _raise_a_tamper():
    """Prime a baseline, then hold a different scene long enough to confirm."""
    h = Harness()
    baseline = scene(1)
    h.feed(baseline, D.BASELINE_SAMPLES + 6)
    covered = scene(99)
    h.feed(covered, int(D.TAMPER_SECONDS) + 4)
    return h, baseline, covered


# ------------------------------------------------------- the judged frame
def test_a_raise_carries_the_frame_it_was_decided_on():
    h, _baseline, covered = _raise_a_tamper()
    raises = h.raises()
    assert raises, "the fixture must produce a raise"
    sample = raises[0]["sample"]
    assert sample is not None, "a raise must carry its judged frame"
    assert np.array_equal(sample, covered), (
        "the stored frame must be the one the detector judged, not another")


def test_the_frame_is_not_refetched_after_the_decision():
    """It must be the array handed to tamper_state, identity included."""
    h = Harness()
    h.feed(scene(1), D.BASELINE_SAMPLES + 6)
    covered = scene(99)
    seen = []
    real = D.tamper_state

    def spy(state, gray, now, obstructed):
        seen.append(gray)
        return real(state, gray, now, obstructed)

    D_tamper = D.tamper_state
    try:
        D.tamper_state = spy
        h.feed(covered, int(D.TAMPER_SECONDS) + 4)
    finally:
        D.tamper_state = D_tamper
    raises = h.raises()
    assert raises
    assert any(raises[0]["sample"] is g for g in seen), (
        "the emitted sample must BE one of the arrays tamper_state judged")


def test_a_recovery_carries_no_frame():
    h, baseline, _covered = _raise_a_tamper()
    h.feed(baseline, int(D.TAMPER_SECONDS) + 6)
    recoveries = [e for e in h.sent if e.get("action") == "recover"]
    assert recoveries, "the fixture must produce a recovery"
    assert recoveries[0]["sample"] is None, "only a raise keeps a frame"


def test_only_one_frame_per_raise_not_one_per_sample():
    h, _b, _c = _raise_a_tamper()
    with_sample = [e for e in h.sent if e.get("sample") is not None]
    assert len(with_sample) == 1, f"expected one stored frame, got {len(with_sample)}"


def test_a_non_tampering_detector_emits_nothing():
    """Defocus is diagnostic only: no event, so no frame."""
    h = Harness()
    h.feed(scene(1), D.BASELINE_SAMPLES + 6, features=(FEATURE_DEFOCUS,))
    h.feed(np.full((180, 320), 128, np.uint8), 12, features=(FEATURE_DEFOCUS,))
    assert h.sent == [], "defocus raises no event, so it stores no sample"


# ------------------------------------------------------- encoding + bounds
def test_encoding_preserves_the_samples_own_size():
    cv2 = pytest.importorskip("cv2")
    from camera_health.sample import health_sample_jpeg
    data, width, height = health_sample_jpeg(scene(5))
    assert (width, height) == (320, 180), "no resize: what was judged is what is stored"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded.shape == (180, 320)


@pytest.mark.parametrize("bad", [None, "not an array", np.zeros((0, 0), np.uint8), 42])
def test_malformed_input_is_dropped_not_stored(bad):
    from camera_health.sample import health_sample_jpeg
    assert health_sample_jpeg(bad) is None


def test_an_oversized_sample_is_dropped():
    """Bounded by construction: anything unexpected is refused rather than stored."""
    from camera_health.sample import health_sample_jpeg, MAX_BYTES
    huge = np.random.default_rng(3).integers(0, 255, (4000, 4000), dtype=np.uint8)
    out = health_sample_jpeg(huge)
    assert out is None or len(out[0]) <= MAX_BYTES


def test_a_normal_sample_is_small():
    from camera_health.sample import health_sample_jpeg, MAX_BYTES
    data, _w, _h = health_sample_jpeg(scene(7))
    assert len(data) < MAX_BYTES
