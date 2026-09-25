"""
Lighting-transition guard (2026-09-19). CPU only, no camera, no network.

    aienv/bin/python test_camera_tamper_lighting.py

Uses REAL CAM-R25 frames (test_fixtures/camera_tamper_lighting/, production
320x180 samples): the 2026-09-19 05:40 lights-on transition that raised a false
Camera Tampering incident (EVT-011BCB49), the 2026-09-17 lights-off transition,
the genuine interference tests P2/P3/P4, people walking and normal footage.
Each case primes the production CameraHealthManager on the "before" frame and
then holds the "after" frame long enough for the 8 s persistence to decide.
"""
import os
import sys
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from camera_health import detectors as D                                   # noqa: E402
from camera_health.manager import (FEATURE_OBSTRUCTION, FEATURE_TAMPER,       # noqa: E402
                                   UMBRELLA_EVENT_TYPE)
from camera_health.state import PHASE_READY                                  # noqa: E402
from test_camera_health import blur, covered_dark, other_scene              # noqa: E402
from test_camera_tamper_incident import Harness                             # noqa: E402

FIX = os.path.join(HERE, "test_fixtures", "camera_tamper_lighting")


def frame(name):
    img = cv2.imread(os.path.join(FIX, name + ".png"), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (180, 320), name
    return img


def run(before, after, hold=30):
    h = Harness()
    h.prime(frame(before) if isinstance(before, str) else before, D.BASELINE_SAMPLES + 5)
    h.feed(frame(after) if isinstance(after, str) else after, hold)
    return h


class GuardConfigTests(unittest.TestCase):
    def test_named_defaults(self):
        if "CAMERA_HEALTH_STRUCTURAL_OVERLAP_LIGHTING_MIN" not in os.environ:
            self.assertEqual(D.STRUCTURAL_OVERLAP_LIGHTING_MIN, 0.80)
        if "CAMERA_HEALTH_LIGHTING_MIN_BRIGHTNESS_RATIO" not in os.environ:
            self.assertEqual(D.LIGHTING_MIN_BRIGHTNESS_RATIO, 1.3)
        self.assertEqual(D.TAMPER_DISTANCE, 0.65, "scene-change threshold NOT raised")

    def test_the_real_lighting_frames_breach_the_tamper_threshold(self):
        """The guard is needed: the unchanged detector would call these tampering."""
        for before, after in (("n3_before", "n3_after"), ("n2_before", "n2_after")):
            d = D.scene_distance(D.scene_signature(frame(before)), D.scene_signature(frame(after)))
            self.assertGreaterEqual(d, D.TAMPER_DISTANCE, before)


class LightingTests(unittest.TestCase):
    def test_07_lights_on_is_not_camera_tampering(self):
        h = run("n3_before", "n3_after")
        self.assertEqual(h.sent, [], "the demonstrated 05:40 lights-on must not raise")
        self.assertEqual(h.state.illumination_resets, 1)
        self.assertTrue(any("lighting transition" in l for l in h.log))

    def test_11_lights_off_is_not_camera_tampering(self):
        h = run("n2_before", "n2_after")
        self.assertEqual(h.sent, [])
        self.assertEqual(h.state.illumination_resets, 1)

    def test_11c_lights_off_with_auto_exposure_flash_is_not_tampering(self):
        """Real 21:19 sequence shape: black, over-exposed, black, then the dim
        settled view. The over-exposed frame reads as 'clear' (small distance)
        and must not become the guard's edge reference."""
        before = frame("n2_before")
        flash = np.clip(before.astype(np.float32) * 1.65, 0, 255).astype(np.uint8)
        black = (before * 0.04).astype(np.uint8)
        h = Harness()
        h.prime(before, D.BASELINE_SAMPLES + 5)
        for img in (black, flash, black):
            h.feed(img, 1)
        h.feed(frame("n2_after"), 30)
        self.assertEqual(h.sent, [], "lights-off with an auto-exposure flash must not raise")
        self.assertEqual(h.state.illumination_resets, 1)

    def test_08_light_transition_rebases_the_baseline(self):
        h = run("n3_before", "n3_after")
        st = h.state
        self.assertEqual(st.scene_phase, PHASE_READY, "baseline rebuilt on the lit view")
        self.assertFalse(st.tamper.active)
        lit = D.scene_signature(frame("n3_after"))
        self.assertLess(D.scene_distance(lit, st.scene_baseline), 0.1)
        # ...and a genuine interference AFTER the lights came on is still caught.
        h.feed(other_scene(9), 12)
        self.assertEqual([(e["action"], e["metadata"]["reason"]) for e in h.sent],
                         [("raise", FEATURE_TAMPER)])

    def test_09_real_physical_covers_still_raise(self):
        for case in ("p2", "p3", "p4"):
            h = run(f"{case}_before", f"{case}_after")
            self.assertEqual([(e["action"], e["metadata"]["reason"], e["event_type"]) for e in h.sent],
                             [("raise", FEATURE_TAMPER, UMBRELLA_EVENT_TYPE)], case)
            self.assertEqual(h.state.illumination_resets, 0, case)

    def test_10_obstruction_still_raises_in_the_dark_and_in_the_light(self):
        for base in ("n3_before", "n3_after"):
            h = run(base, covered_dark(frame(base)))
            self.assertEqual([(e["action"], e["metadata"]["reason"]) for e in h.sent],
                             [("raise", FEATURE_OBSTRUCTION)], base)
            self.assertEqual(h.state.illumination_resets, 0, base)

    def test_obstruction_candidate_blocks_the_guard(self):
        h = run("n3_before", "n3_before", hold=1)
        h.state.obstruction.since = h.t                      # obstruction candidate open
        detail = D.tamper_state(h.state, frame("n3_after"), h.t, False)[1]
        self.assertNotIn("illumination_transition", detail)
        self.assertEqual(h.state.illumination_resets, 0)

    def test_11b_people_walking_is_not_treated_as_lighting(self):
        ok, detail = D.lighting_transition(run("n1_before", "n1_before", hold=1).state, frame("n1_after"))
        self.assertFalse(ok, detail)
        # and a people-like change that PERSISTS still reaches the normal decision
        h = run("n1_before", "n1_after")
        self.assertEqual([e["action"] for e in h.sent], ["raise"])
        self.assertEqual(h.state.illumination_resets, 0)

    def test_normal_footage_is_unaffected(self):
        h = run("day_a", "day_b")
        self.assertEqual(h.sent, [])
        self.assertEqual(h.state.illumination_resets, 0)

    def test_12_defocus_is_not_suppressed_and_is_still_not_tampering(self):
        h = run("n3_after", blur(frame("n3_after"), 25), hold=60)
        self.assertTrue(h.state.defocus.active, "defocus still detected (diagnostic)")
        self.assertEqual(h.sent, [])
        self.assertEqual(h.state.illumination_resets, 0)

    def test_13_signal_loss_blocks_the_guard_and_is_still_not_tampering(self):
        h = run("n3_before", "n3_before", hold=1)
        h.state.signal.active = True
        detail = D.tamper_state(h.state, frame("n3_after"), h.t, False)[1]
        self.assertNotIn("illumination_transition", detail)
        self.assertEqual(h.state.illumination_resets, 0)

    def test_active_incident_is_never_rebaselined_by_the_guard(self):
        """Re-baselining a raised incident would fake a recovery."""
        h = run("n3_before", other_scene(9), hold=12)
        self.assertEqual([e["action"] for e in h.sent], ["raise"])
        h.feed(frame("n3_after"), 30)
        self.assertEqual([e["action"] for e in h.sent], ["raise"], "no fake recovery")
        self.assertEqual(h.state.illumination_resets, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
