"""Sharpness-aware lighting guard (2026-10-10). CPU only, no camera, no network.

THE REGRESSION: brightness and structural overlap alone cannot separate "the lights changed"
from "a hand is over the lens near a window". On the 2026-09-30 camera_03 cover both measured
like a lighting change (brightness 1.31-2.45, overlap 0.83-0.99), so the guard called it
illumination, re-learned its baselines on the COVERED view, and discarded a genuine tamper
candidate at 6.0 s of the required 8.0 s.

Sharpness separates them because the metric is lighting-normalised by construction
(metrics.py: "ratio 1.01 at 28% gain" for re-lighting, "<= 0.08 of baseline" for a collapse).

FIXTURES. The original camera_tamper_lighting PNGs were never committed to this repository -
`git rev-list --objects --all` finds no PNG blob in any tree - so those historical cases cannot
be replayed here. These tests therefore use:
  * test_fixtures/camera_tamper_sharpness/*.png - REAL 320x180 samples cut from
    tamper/camera_03_20260930_160800_160900.mp4 (t=17 s clean, t=26 s covered), the very
    footage the guard was mishandling; and
  * a synthetic pure-gain frame for the negative case, which is sound precisely because
    sharpness is lighting-normalised: a gain change is the one thing it must ignore.
"""
import os
import sys
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from camera_health import detectors as D                                      # noqa: E402
from camera_health import metrics as M                                        # noqa: E402
from camera_health.state import CameraState                                   # noqa: E402

FIX = os.path.join(HERE, "test_fixtures", "camera_tamper_sharpness")


def fixture(name):
    img = cv2.imread(os.path.join(FIX, name + ".png"), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (180, 320), name
    return img


def primed_state(baseline_img, camera="camera_03"):
    """A state whose sharpness baseline is learned from `baseline_img`."""
    state = CameraState(camera, 0.0)
    for _ in range(D.BASELINE_SAMPLES):
        D.defocus_state(state, M.sharpness(baseline_img), obstructed=False)
    assert state.sharpness_baseline is not None, "baseline failed to establish"
    return state


class MeasuredSeparationTests(unittest.TestCase):
    """The numbers the threshold was chosen from, asserted rather than asserted-about."""

    def test_the_real_cover_collapses_sharpness(self):
        clean, covered = fixture("c03_clean"), fixture("c03_covered")
        ratio = M.sharpness(covered) / M.sharpness(clean)
        self.assertLess(ratio, 0.1, f"real cover sharpness ratio {ratio:.4f}")

    def test_a_pure_gain_change_does_not(self):
        """The metric is lighting-normalised: this is what makes it a usable discriminator."""
        clean = fixture("c03_clean")
        relit = np.clip(clean.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
        ratio = M.sharpness(relit) / M.sharpness(clean)
        self.assertGreater(ratio, 0.5, f"gain change moved sharpness to {ratio:.4f}")

    def test_the_two_cases_sit_either_side_of_the_floor(self):
        clean = fixture("c03_clean")
        relit = np.clip(clean.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
        cover_ratio = M.sharpness(fixture("c03_covered")) / M.sharpness(clean)
        relit_ratio = M.sharpness(relit) / M.sharpness(clean)
        self.assertLess(cover_ratio, D.LIGHTING_SHARPNESS_COLLAPSE_RATIO)
        self.assertGreater(relit_ratio, D.LIGHTING_SHARPNESS_COLLAPSE_RATIO)


class LightingSharpnessCredibleTests(unittest.TestCase):

    def test_a_cover_makes_the_lighting_story_incredible(self):
        state = primed_state(fixture("c03_clean"))
        ok, detail = D.lighting_sharpness_credible(state, fixture("c03_covered"))
        self.assertFalse(ok, detail)
        self.assertLess(detail["sharpness_ratio"], D.LIGHTING_SHARPNESS_COLLAPSE_RATIO)

    def test_a_gain_change_keeps_it_credible(self):
        clean = fixture("c03_clean")
        state = primed_state(clean)
        relit = np.clip(clean.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
        ok, detail = D.lighting_sharpness_credible(state, relit)
        self.assertTrue(ok, detail)

    def test_no_baseline_does_not_switch_the_guard_off(self):
        """Missing data must never be what disables a safeguard."""
        state = CameraState("camera_03", 0.0)
        self.assertIsNone(state.sharpness_baseline)
        ok, detail = D.lighting_sharpness_credible(state, fixture("c03_covered"))
        self.assertTrue(ok, "an unlearned baseline must leave the guard exactly as it was")
        self.assertEqual(detail.get("sharpness_check"), "no baseline")

    def test_a_degenerate_baseline_does_not_switch_the_guard_off(self):
        state = CameraState("camera_03", 0.0)
        state.sharpness_baseline = 0.0
        ok, _ = D.lighting_sharpness_credible(state, fixture("c03_covered"))
        self.assertTrue(ok, "a zero baseline must not be treated as a collapse")


class GuardIntegrationTests(unittest.TestCase):
    """What the guard DOES with the verdict, which is the part that bit us.

    `c03_cover_looks_like_lighting.png` is the real t=28 s sample from the 2026-09-30 cover -
    the one the OLD guard excused. Measured against the t=17 s clean baseline, every sample the
    old guard would have excused sits below the new floor:

        t    distance  brightness  overlap  sharpness_ratio   old guard
        28     1.552      1.310      0.845       0.135         EXCUSED
        29     1.458      1.379      0.848       0.192         EXCUSED
        31     0.931      1.454      0.994       0.491         EXCUSED
        33     1.498      2.448      0.824       0.110         EXCUSED
        34     0.152      1.177      0.944       1.075         (view restored)
    """

    def _ready_state(self, img):
        state = primed_state(img)
        for _ in range(D.SCENE_WARMUP_SAMPLES + D.SCENE_BASELINE_SAMPLES + 2):
            D.tamper_state(state, img, 0.0, obstructed=False)
        return state

    def test_the_old_guard_really_would_have_excused_this_frame(self):
        """Without this, the test below would prove nothing."""
        state = self._ready_state(fixture("c03_clean"))
        ok, check = D.lighting_transition(state, fixture("c03_cover_looks_like_lighting"))
        self.assertTrue(ok, f"brightness+overlap alone no longer excuse the cover: {check}")

    def test_sharpness_is_what_flips_the_verdict(self):
        state = self._ready_state(fixture("c03_clean"))
        credible, detail = D.lighting_sharpness_credible(
            state, fixture("c03_cover_looks_like_lighting"))
        self.assertFalse(credible, detail)
        self.assertLess(detail["sharpness_ratio"], D.LIGHTING_SHARPNESS_COLLAPSE_RATIO)

    def test_a_cover_no_longer_rebaselines_the_scene(self):
        """THE BUG: reset_image_baselines() learned the COVERED view as the normal scene,
        after which the tamper could never raise."""
        state = self._ready_state(fixture("c03_clean"))
        before_resets = state.illumination_resets
        baseline_before = state.scene_baseline
        _, detail = D.tamper_state(
            state, fixture("c03_cover_looks_like_lighting"), 10.0, obstructed=False)
        self.assertEqual(state.illumination_resets, before_resets,
                         f"the cover was still excused as a lighting transition: {detail}")
        self.assertNotIn("illumination_transition", detail)
        self.assertIsNotNone(state.scene_baseline, "scene baseline was wiped by the cover")
        self.assertTrue(np.array_equal(state.scene_baseline, baseline_before),
                        "the covered view replaced the baseline - the exact contamination "
                        "this change exists to prevent")

    def test_the_rejection_is_recorded_for_an_operator(self):
        """A safeguard that changes its mind silently is not diagnosable."""
        state = self._ready_state(fixture("c03_clean"))
        _, detail = D.tamper_state(
            state, fixture("c03_cover_looks_like_lighting"), 10.0, obstructed=False)
        check = detail.get("lighting_check") or {}
        self.assertEqual(check.get("lighting_rejected"), "sharpness collapsed", detail)
        self.assertIn("sharpness_ratio", check)

    def test_a_restored_view_is_still_treated_as_lighting(self):
        """t=34: structure back (0.944) and sharpness recovered (1.075). The guard must
        still be allowed to re-baseline here, or it would never recover from a real relight."""
        clean = fixture("c03_clean")
        state = self._ready_state(clean)
        credible, detail = D.lighting_sharpness_credible(state, clean)
        self.assertTrue(credible, detail)
        self.assertGreaterEqual(detail["sharpness_ratio"], D.LIGHTING_SHARPNESS_COLLAPSE_RATIO)


class UnchangedDefaultsTests(unittest.TestCase):
    """Nothing else was loosened to make the above work."""

    def test_existing_thresholds_are_untouched(self):
        if "CAMERA_HEALTH_TAMPER_SECONDS" not in os.environ:
            self.assertEqual(D.TAMPER_SECONDS, 8.0)
        if "CAMERA_HEALTH_STRUCTURAL_OVERLAP_LIGHTING_MIN" not in os.environ:
            self.assertEqual(D.STRUCTURAL_OVERLAP_LIGHTING_MIN, 0.80)
        if "CAMERA_HEALTH_LIGHTING_MIN_BRIGHTNESS_RATIO" not in os.environ:
            self.assertEqual(D.LIGHTING_MIN_BRIGHTNESS_RATIO, 1.3)
        self.assertEqual(D.TAMPER_DISTANCE, 0.65)
        self.assertEqual(D.DEFOCUS_RATIO, 0.50)
        self.assertEqual(D.OBSTRUCTION_SECONDS, 5.0)
        self.assertEqual(D.SIGNAL_LOSS_SECONDS, 120.0)

    def test_the_floor_is_configurable_and_sits_below_the_defocus_bound(self):
        """ZAYED 2026-10-12: 0.5 was chosen to be the same bound as DEFOCUS_RATIO - one notion
        of "collapsed". Replaying the real clips disproved the assumption behind it: a room
        going dark takes sharpness to 0.465-0.476, under that floor, so the guard overrode
        structure_intact on a light switch. 0.35 sits in the measured gap between the covers
        (<= 0.271) and the lights-off (>= 0.465). This guard OVERRIDES a safeguard, so it must
        be harder to trip than plain defocus, not equal to it."""
        if "CAMERA_HEALTH_LIGHTING_SHARPNESS_COLLAPSE_RATIO" not in os.environ:
            self.assertEqual(D.LIGHTING_SHARPNESS_COLLAPSE_RATIO, 0.35)
            self.assertLess(D.LIGHTING_SHARPNESS_COLLAPSE_RATIO, D.DEFOCUS_RATIO,
                            "overriding a safeguard must be stricter than detecting defocus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
