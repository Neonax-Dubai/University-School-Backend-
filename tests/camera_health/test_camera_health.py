"""Unit tests for camera_health. CPU only, no camera, no network, no database."""

import sys, time, unittest
import numpy as np
import cv2

sys.path.insert(0, "/home/matrix/Dubai_Police/AI_inferencing")
from camera_health import metrics as M
from camera_health import detectors as D
from camera_health.manager import (UMBRELLA_EVENT_TYPE,
                                   CameraHealthManager, FEATURE_SIGNAL_LOSS,
                                   FEATURE_OBSTRUCTION, FEATURE_DEFOCUS,
                                   FEATURE_TAMPER, CAMERA_HEALTH_FEATURES)
from camera_health.state import CameraState

RNG = np.random.default_rng(7)


def scene(seed=0, w=M.SAMPLE_WIDTH, h=M.SAMPLE_HEIGHT):
    """A structured synthetic scene: edges, texture and gradient, like a room."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.uint8)
    img[:] = np.linspace(60, 190, w, dtype=np.uint8)[None, :]
    for i in range(14):
        x = int(rng.integers(0, w - 30)); y = int(rng.integers(0, h - 24))
        img[y:y + int(rng.integers(8, 24)), x:x + int(rng.integers(8, 30))] = int(rng.integers(0, 255))
    img = np.clip(img.astype(np.int16) + rng.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)
    return img


def other_scene(seed=0, w=M.SAMPLE_WIDTH, h=M.SAMPLE_HEIGHT):
    """
    A structurally DIFFERENT scene, not just a different seed.

    scene() lays a horizontal brightness ramp under its blocks, so two scene()
    images share that dominant structure and score only 0.20 apart - well below
    the 0.65 threshold measured from real cameras. Rotating the ramp is what
    makes this a different view rather than the same view rearranged.
    """
    return np.ascontiguousarray(np.rot90(scene(seed, w=h, h=w)))


def dim(img, gain):
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)


def covered_dark(img):
    return np.clip(img.astype(np.float32) * 0.05, 0, 255).astype(np.uint8)


def covered_white(img):
    return np.clip(img.astype(np.float32) * 0.04 + 240, 0, 255).astype(np.uint8)


class FakeReader:
    def __init__(self, frames_read=0):
        self.frames_read = frames_read


def manager(shadow=True, features=CAMERA_HEALTH_FEATURES, cameras=("CAM-A",)):
    m = CameraHealthManager(shadow=shadow, logger=lambda line: None)
    m.set_enabled_cameras({c: {f: True for f in features} for c in cameras})
    return m


def feed(m, camera, img, start, count, step=1.0, features=None):
    """Push `count` samples one second apart and return the actions seen."""
    seen = []
    for i in range(count):
        t = start + i * step
        res = m.process_sample(camera, img if not callable(img) else img(i), t, features)
        for feature, (action, detail) in (res or {}).items():
            if action:
                seen.append((t, feature, action))
    return seen


def prime(m, camera, img, start, samples=D.BASELINE_SAMPLES + 2, features=None):
    """Establish this camera's baselines on a healthy scene.

    Long enough for BOTH baselines: the sharpness baseline needs
    BASELINE_SAMPLES, and the scene baseline needs SCENE_WARMUP_SAMPLES
    discarded plus SCENE_BASELINE_SAMPLES mutually stable ones after that.
    """
    feed(m, camera, img, start, samples, features=features)
    return start + samples


# ============================================================
# SIGNAL LOSS
# ============================================================

class SignalLossTests(unittest.TestCase):
    def test_normal_frames_raise_nothing(self):
        m = manager(features=[FEATURE_SIGNAL_LOSS])
        r = FakeReader(0)
        for i in range(20):
            r.frames_read += 25
            m.note_streams({"CAM-A": r}, now=1000.0 + i)
        self.assertEqual(m.transitions, 0)

    def test_brief_gap_is_not_an_outage(self):
        m = manager(features=[FEATURE_SIGNAL_LOSS])
        r = FakeReader(100)
        m.note_streams({"CAM-A": r}, now=1000.0)
        m.note_streams({"CAM-A": r}, now=1003.9)      # 3.9s < 5.0s threshold
        self.assertEqual(m.transitions, 0)
        r.frames_read += 30
        m.note_streams({"CAM-A": r}, now=1004.5)
        self.assertEqual(m.transitions, 0)

    def test_sustained_gap_raises_once(self):
        m = manager(features=[FEATURE_SIGNAL_LOSS])
        r = FakeReader(100)
        m.note_streams({"CAM-A": r}, now=1000.0)
        for t in (1006.0, 1007.0, 1010.0, 1030.0, 1060.0):
            m.note_streams({"CAM-A": r}, now=t)
        st = m._states["CAM-A"]
        self.assertTrue(st.signal.active)
        self.assertEqual(st.signal.raises, 1, "one outage must be one event")

    def test_recovery_after_outage(self):
        m = manager(features=[FEATURE_SIGNAL_LOSS])
        r = FakeReader(100)
        m.note_streams({"CAM-A": r}, now=1000.0)
        m.note_streams({"CAM-A": r}, now=1006.0)
        st = m._states["CAM-A"]
        self.assertTrue(st.signal.active)
        r.frames_read += 10
        m.note_streams({"CAM-A": r}, now=1007.0)
        self.assertFalse(st.signal.active)
        self.assertEqual(st.signal.recoveries, 1)

    def test_no_duplicate_events_while_offline(self):
        m = manager(features=[FEATURE_SIGNAL_LOSS])
        r = FakeReader(100)
        m.note_streams({"CAM-A": r}, now=1000.0)
        for i in range(200):
            m.note_streams({"CAM-A": r}, now=1006.0 + i)
        self.assertEqual(m._states["CAM-A"].signal.raises, 1)

    def test_cooldown_blocks_immediate_reraise(self):
        m = manager(features=[FEATURE_SIGNAL_LOSS])
        r = FakeReader(100)
        m.note_streams({"CAM-A": r}, now=1000.0)
        m.note_streams({"CAM-A": r}, now=1006.0)       # raise 1
        r.frames_read += 5
        m.note_streams({"CAM-A": r}, now=1007.0)       # recover
        m.note_streams({"CAM-A": r}, now=1013.0)       # gap again, inside cooldown
        self.assertEqual(m._states["CAM-A"].signal.raises, 1)

    def test_feature_off_means_no_signal_tracking(self):
        m = manager(features=[FEATURE_DEFOCUS])
        r = FakeReader(100)
        m.note_streams({"CAM-A": r}, now=1000.0)
        m.note_streams({"CAM-A": r}, now=1099.0)
        self.assertEqual(m.transitions, 0)


# ============================================================
# OBSTRUCTION
# ============================================================

class ObstructionTests(unittest.TestCase):
    def test_normal_image_is_clear(self):
        m = manager(features=[FEATURE_OBSTRUCTION])
        feed(m, "CAM-A", scene(1), 0.0, 40)
        self.assertFalse(m._states["CAM-A"].obstruction.active)

    def test_dark_but_structured_night_scene_is_not_obstruction(self):
        m = manager(features=[FEATURE_OBSTRUCTION])
        t = prime(m, "CAM-A", scene(2), 0.0)
        feed(m, "CAM-A", dim(scene(2), 0.28), t, 30)
        self.assertFalse(m._states["CAM-A"].obstruction.active,
                         "a dim scene that keeps its edges must not be obstruction")

    def test_sustained_covered_lens_raises(self):
        m = manager(features=[FEATURE_OBSTRUCTION])
        t = prime(m, "CAM-A", scene(3), 0.0)
        feed(m, "CAM-A", covered_dark(scene(3)), t, 10)
        st = m._states["CAM-A"]
        self.assertTrue(st.obstruction.active)
        self.assertEqual(st.obstruction.raises, 1)

    def test_white_covered_lens_raises(self):
        m = manager(features=[FEATURE_OBSTRUCTION])
        t = prime(m, "CAM-A", scene(4), 0.0)
        feed(m, "CAM-A", covered_white(scene(4)), t, 10)
        self.assertTrue(m._states["CAM-A"].obstruction.active)

    def test_brief_obstruction_does_not_raise(self):
        m = manager(features=[FEATURE_OBSTRUCTION])
        t = prime(m, "CAM-A", scene(5), 0.0)
        feed(m, "CAM-A", covered_dark(scene(5)), t, 3)      # 3s < 5s persistence
        self.assertFalse(m._states["CAM-A"].obstruction.active)

    def test_recovery(self):
        m = manager(features=[FEATURE_OBSTRUCTION])
        t = prime(m, "CAM-A", scene(6), 0.0)
        feed(m, "CAM-A", covered_dark(scene(6)), t, 10)
        st = m._states["CAM-A"]
        self.assertTrue(st.obstruction.active)
        # A recovery now waits the SAME window as the raise (Condition
        # .update recover_persistence), so a raise and its recovery can no
        # longer land a second apart - see test_camera_health_recovery.py.
        feed(m, "CAM-A", scene(6), t + 10, int(D.OBSTRUCTION_SECONDS) + 3)
        self.assertFalse(st.obstruction.active)
        self.assertEqual(st.obstruction.recoveries, 1)

    def test_one_event_per_obstruction(self):
        m = manager(features=[FEATURE_OBSTRUCTION])
        t = prime(m, "CAM-A", scene(7), 0.0)
        feed(m, "CAM-A", covered_dark(scene(7)), t, 120)
        self.assertEqual(m._states["CAM-A"].obstruction.raises, 1)


# ============================================================
# DEFOCUS
# ============================================================

class DefocusTests(unittest.TestCase):
    def test_sharp_image_is_clear(self):
        m = manager(features=[FEATURE_DEFOCUS])
        t = prime(m, "CAM-A", scene(11), 0.0)
        feed(m, "CAM-A", scene(11), t, 20)
        self.assertFalse(m._states["CAM-A"].defocus.active)

    def test_strong_blur_raises(self):
        m = manager(features=[FEATURE_DEFOCUS])
        t = prime(m, "CAM-A", scene(12), 0.0)
        feed(m, "CAM-A", blur(scene(12), 15), t, 10)
        st = m._states["CAM-A"]
        self.assertTrue(st.defocus.active)
        self.assertEqual(st.defocus.raises, 1)

    def test_mild_blur_raises(self):
        m = manager(features=[FEATURE_DEFOCUS])
        t = prime(m, "CAM-A", scene(13), 0.0)
        feed(m, "CAM-A", blur(scene(13), 9), t, 10)
        self.assertTrue(m._states["CAM-A"].defocus.active)

    def test_brief_blur_does_not_raise(self):
        m = manager(features=[FEATURE_DEFOCUS])
        t = prime(m, "CAM-A", scene(14), 0.0)
        feed(m, "CAM-A", blur(scene(14), 15), t, 3)
        self.assertFalse(m._states["CAM-A"].defocus.active)

    def test_lighting_change_is_not_defocus(self):
        """The measured failure of raw Laplacian variance, guarded."""
        m = manager(features=[FEATURE_DEFOCUS])
        t = prime(m, "CAM-A", scene(15), 0.0)
        feed(m, "CAM-A", dim(scene(15), 0.28), t, 30)
        self.assertFalse(m._states["CAM-A"].defocus.active,
                         "a dimmer scene is not an out-of-focus scene")

    def test_recovery(self):
        m = manager(features=[FEATURE_DEFOCUS])
        t = prime(m, "CAM-A", scene(16), 0.0)
        feed(m, "CAM-A", blur(scene(16), 15), t, 10)
        st = m._states["CAM-A"]
        self.assertTrue(st.defocus.active)
        # A recovery now waits the SAME window as the raise (Condition
        # .update recover_persistence), so a raise and its recovery can no
        # longer land a second apart - see test_camera_health_recovery.py.
        feed(m, "CAM-A", scene(16), t + 10, int(D.DEFOCUS_SECONDS) + 3)
        self.assertFalse(st.defocus.active)
        self.assertEqual(st.defocus.recoveries, 1)

    def test_no_baseline_means_no_decision(self):
        m = manager(features=[FEATURE_DEFOCUS])
        feed(m, "CAM-A", blur(scene(17), 15), 0.0, 8)
        self.assertFalse(m._states["CAM-A"].defocus.active)

    def test_covered_lens_does_not_also_raise_defocus(self):
        m = manager(features=[FEATURE_OBSTRUCTION, FEATURE_DEFOCUS])
        t = prime(m, "CAM-A", scene(18), 0.0)
        feed(m, "CAM-A", covered_dark(scene(18)), t, 20)
        st = m._states["CAM-A"]
        self.assertTrue(st.obstruction.active)
        self.assertFalse(st.defocus.active, "one fault must not raise two events")


# ============================================================
# TAMPER
# ============================================================

class TamperTests(unittest.TestCase):
    def test_normal_scene_is_clear(self):
        m = manager(features=[FEATURE_TAMPER])
        t = prime(m, "CAM-A", scene(21), 0.0)
        feed(m, "CAM-A", lambda i: scene(21), t, 20)
        self.assertFalse(m._states["CAM-A"].tamper.active)

    def test_people_moving_is_not_tamper(self):
        m = manager(features=[FEATURE_TAMPER])
        base = scene(22)
        def with_person(i):
            img = base.copy()
            x = 10 + (i * 9) % (M.SAMPLE_WIDTH - 30)
            img[70:150, x:x + 22] = 30
            return img
        t = prime(m, "CAM-A", with_person, 0.0)
        feed(m, "CAM-A", with_person, t, 40)
        self.assertFalse(m._states["CAM-A"].tamper.active)

    def test_persistent_scene_change_raises(self):
        m = manager(features=[FEATURE_TAMPER])
        t = prime(m, "CAM-A", scene(23), 0.0)
        feed(m, "CAM-A", other_scene(99), t, 20)
        st = m._states["CAM-A"]
        self.assertTrue(st.tamper.active)
        self.assertEqual(st.tamper.raises, 1)

    def test_camera_rotated_raises(self):
        m = manager(features=[FEATURE_TAMPER])
        base = scene(24)
        t = prime(m, "CAM-A", base, 0.0)
        feed(m, "CAM-A", np.roll(base, 160, axis=1), t, 20)
        self.assertTrue(m._states["CAM-A"].tamper.active)

    def test_temporary_change_does_not_raise(self):
        m = manager(features=[FEATURE_TAMPER])
        t = prime(m, "CAM-A", scene(25), 0.0)
        feed(m, "CAM-A", other_scene(98), t, 5)             # 5s < 10s persistence
        self.assertFalse(m._states["CAM-A"].tamper.active)

    def test_lighting_change_is_not_tamper(self):
        m = manager(features=[FEATURE_TAMPER])
        t = prime(m, "CAM-A", scene(26), 0.0)
        feed(m, "CAM-A", dim(scene(26), 0.55), t, 30)
        self.assertFalse(m._states["CAM-A"].tamper.active,
                         "a lighting change must not read as a moved camera")

    def test_return_to_baseline_recovers(self):
        m = manager(features=[FEATURE_TAMPER])
        t = prime(m, "CAM-A", scene(27), 0.0)
        feed(m, "CAM-A", other_scene(97), t, 20)
        st = m._states["CAM-A"]
        self.assertTrue(st.tamper.active)
        # A recovery now waits the SAME window as the raise (Condition
        # .update recover_persistence), so a raise and its recovery can no
        # longer land a second apart - see test_camera_health_recovery.py.
        feed(m, "CAM-A", scene(27), t + 20, int(D.TAMPER_SECONDS) + 3)
        self.assertFalse(st.tamper.active)
        self.assertEqual(st.tamper.recoveries, 1)

    def test_baseline_is_frozen_while_tampered(self):
        """Otherwise the tampered view silently becomes the new normal."""
        m = manager(features=[FEATURE_TAMPER])
        t = prime(m, "CAM-A", scene(28), 0.0)
        before = m._states["CAM-A"].scene_baseline.copy()
        feed(m, "CAM-A", other_scene(96), t, 60)
        after = m._states["CAM-A"].scene_baseline
        self.assertTrue(np.allclose(before, after, atol=1e-6),
                        "baseline must not drift toward the tampered scene")


# ============================================================
# QUEUE
# ============================================================

class QueueTests(unittest.TestCase):
    def _frame(self):
        return cv2.cvtColor(scene(31), cv2.COLOR_GRAY2BGR)

    def test_one_pending_sample_per_camera(self):
        m = manager()
        f = self._frame()
        for i in range(50):
            m.observe("CAM-A", f, timestamp=1000.0 + i * 2)
        self.assertLessEqual(len(m._pending), 1)
        self.assertEqual(m.stats()["queue_max_per_camera"], 1)

    def test_newest_sample_replaces_old(self):
        m = manager()
        a = cv2.cvtColor(np.full((M.SAMPLE_HEIGHT, M.SAMPLE_WIDTH), 10, np.uint8), cv2.COLOR_GRAY2BGR)
        b = cv2.cvtColor(np.full((M.SAMPLE_HEIGHT, M.SAMPLE_WIDTH), 200, np.uint8), cv2.COLOR_GRAY2BGR)
        m.observe("CAM-A", a, timestamp=1000.0)
        m.observe("CAM-A", b, timestamp=1002.0)
        gray, ts = m._pending["CAM-A"]
        self.assertEqual(ts, 1002.0)
        self.assertGreater(float(gray.mean()), 100.0, "the newest sample must win")
        self.assertEqual(m.samples_dropped, 1)

    def test_queue_never_exceeds_camera_count(self):
        cams = [f"CAM-{i}" for i in range(12)]
        m = manager(cameras=cams)
        f = self._frame()
        for i in range(30):
            for c in cams:
                m.observe(c, f, timestamp=1000.0 + i * 2)
        self.assertLessEqual(len(m._pending), len(cams))

    def test_rate_limit_holds(self):
        m = manager()
        f = self._frame()
        taken = sum(1 for i in range(100) if m.observe("CAM-A", f, timestamp=1000.0 + i * 0.04))
        self.assertLessEqual(taken, 5, "1 Hz sampling must reject the other frames")

    def test_worker_slower_than_producer_drops_not_grows(self):
        m = CameraHealthManager(shadow=True, logger=lambda l: None)
        m.set_enabled_cameras({"CAM-A": {f: True for f in CAMERA_HEALTH_FEATURES}})
        f = self._frame()
        for i in range(200):
            m.observe("CAM-A", f, timestamp=1000.0 + i * 2)
        self.assertLessEqual(len(m._pending), 1)
        self.assertGreater(m.samples_dropped, 0)

    def test_disabled_camera_is_not_sampled(self):
        m = manager(cameras=["CAM-A"])
        self.assertFalse(m.observe("CAM-B", self._frame(), timestamp=1000.0))

    def test_observe_never_raises(self):
        m = manager()
        self.assertFalse(m.observe("CAM-A", "not a frame", timestamp=1000.0))
        self.assertFalse(m.observe("CAM-A", None, timestamp=1001.0))
        self.assertGreaterEqual(m.observe_errors, 0)


# ============================================================
# MULTI-CAMERA ISOLATION
# ============================================================

class MultiCameraTests(unittest.TestCase):
    def test_one_camera_fault_does_not_touch_another(self):
        m = manager(cameras=["CAM-A", "CAM-B"])
        ta = prime(m, "CAM-A", scene(41), 0.0)
        tb = prime(m, "CAM-B", scene(42), 0.0)
        feed(m, "CAM-A", covered_dark(scene(41)), ta, 10)
        feed(m, "CAM-B", scene(42), tb, 10)
        self.assertTrue(m._states["CAM-A"].obstruction.active)
        self.assertFalse(m._states["CAM-B"].obstruction.active)
        self.assertFalse(m._states["CAM-B"].defocus.active)
        self.assertFalse(m._states["CAM-B"].tamper.active)

    def test_baselines_are_independent(self):
        m = manager(cameras=["CAM-A", "CAM-B"])
        prime(m, "CAM-A", scene(43), 0.0)
        prime(m, "CAM-B", blur(scene(44), 5), 0.0)
        a = m._states["CAM-A"].sharpness_baseline
        b = m._states["CAM-B"].sharpness_baseline
        self.assertIsNotNone(a); self.assertIsNotNone(b)
        self.assertNotAlmostEqual(a, b, places=3)

    def test_signal_loss_is_per_camera(self):
        m = manager(features=[FEATURE_SIGNAL_LOSS], cameras=["CAM-A", "CAM-B"])
        a, b = FakeReader(10), FakeReader(10)
        m.note_streams({"CAM-A": a, "CAM-B": b}, now=1000.0)
        b.frames_read += 30
        m.note_streams({"CAM-A": a, "CAM-B": b}, now=1006.0)
        self.assertTrue(m._states["CAM-A"].signal.active)
        self.assertFalse(m._states["CAM-B"].signal.active)


# ============================================================
# EVENTS / SHADOW MODE
# ============================================================

class EventTests(unittest.TestCase):
    def test_shadow_mode_emits_nothing(self):
        sent = []
        m = CameraHealthManager(shadow=True, logger=lambda l: None,
                                event_sink=lambda **kw: sent.append(kw))
        m.set_enabled_cameras({"CAM-A": {FEATURE_OBSTRUCTION: True}})
        t = prime(m, "CAM-A", scene(51), 0.0, features={FEATURE_OBSTRUCTION})
        feed(m, "CAM-A", covered_dark(scene(51)), t, 10, features={FEATURE_OBSTRUCTION})
        self.assertTrue(m._states["CAM-A"].obstruction.active)
        self.assertEqual(sent, [], "shadow mode must never create an event")

    def test_live_mode_emits_one_event_per_transition(self):
        sent = []
        m = CameraHealthManager(shadow=False, logger=lambda l: None,
                                event_sink=lambda **kw: sent.append(kw))
        m.set_enabled_cameras({"CAM-A": {FEATURE_OBSTRUCTION: True}})
        t = prime(m, "CAM-A", scene(52), 0.0, features={FEATURE_OBSTRUCTION})
        feed(m, "CAM-A", covered_dark(scene(52)), t, 40, features={FEATURE_OBSTRUCTION})
        self.assertEqual(len(sent), 1)
        # ONE event type for every camera-interference detector; which one
        # noticed is metadata["reason"] - see manager.UMBRELLA_EVENT_TYPE.
        self.assertEqual(sent[0]["event_type"], UMBRELLA_EVENT_TYPE)
        self.assertEqual(sent[0]["metadata"]["reason"], FEATURE_OBSTRUCTION)
        self.assertNotIn("recovered", sent[0]["metadata"])

    def test_recovery_uses_the_existing_recovered_contract(self):
        sent = []
        m = CameraHealthManager(shadow=False, logger=lambda l: None,
                                event_sink=lambda **kw: sent.append(kw))
        m.set_enabled_cameras({"CAM-A": {FEATURE_OBSTRUCTION: True}})
        t = prime(m, "CAM-A", scene(53), 0.0, features={FEATURE_OBSTRUCTION})
        feed(m, "CAM-A", covered_dark(scene(53)), t, 10, features={FEATURE_OBSTRUCTION})
        # A recovery now waits the SAME window as the raise (Condition
        # .update recover_persistence), so a raise and its recovery can no
        # longer land a second apart - see test_camera_health_recovery.py.
        feed(m, "CAM-A", scene(53), t + 10, int(D.OBSTRUCTION_SECONDS) + 3,
             features={FEATURE_OBSTRUCTION})
        self.assertEqual(len(sent), 2)
        self.assertIs(sent[1]["metadata"]["recovered"], True)

    def test_no_event_spam_under_flapping(self):
        sent = []
        m = CameraHealthManager(shadow=False, logger=lambda l: None,
                                event_sink=lambda **kw: sent.append(kw))
        m.set_enabled_cameras({"CAM-A": {FEATURE_OBSTRUCTION: True}})
        t = prime(m, "CAM-A", scene(54), 0.0, features={FEATURE_OBSTRUCTION})
        for cycle in range(6):
            t = t + 10
            feed(m, "CAM-A", covered_dark(scene(54)), t, 8, features={FEATURE_OBSTRUCTION})
            t = t + 8
            feed(m, "CAM-A", scene(54), t, 2, features={FEATURE_OBSTRUCTION})
        raises = [s for s in sent if "recovered" not in s["metadata"]]
        self.assertEqual(len(raises), 1, "the 300s cooldown must suppress re-raises")

    def test_event_types_match_the_existing_contract(self):
        for feature in CAMERA_HEALTH_FEATURES:
            self.assertIn(feature, ("camera_signal_loss", "camera_obstruction",
                                    "camera_defocus", "camera_tamper"))


# ============================================================
# TAMPER BASELINE LIFECYCLE  (regression for the live shadow defect)
# ============================================================

class TamperBaselineLifecycleTests(unittest.TestCase):
    """
    A live production run confirmed tamper on 2 of 2 cameras 12 s after start
    (distance 1.2266 / 1.2541 vs threshold 0.65) and never recovered, because
    the baseline was taken from the first decoded sample. These tests hold the
    fix closed. The threshold is NOT part of the fix and stays at 0.65.
    """

    def test_threshold_is_unchanged(self):
        self.assertEqual(D.TAMPER_DISTANCE, 0.65)

    def test_first_frame_noisy_does_not_become_baseline(self):
        m = manager(features=[FEATURE_TAMPER])
        noise = RNG.integers(0, 255, (M.SAMPLE_HEIGHT, M.SAMPLE_WIDTH), dtype=np.uint8)
        good = scene(61)
        m.process_sample("CAM-A", noise, 0.0, {FEATURE_TAMPER})
        feed(m, "CAM-A", good, 1.0, 40, features={FEATURE_TAMPER})
        st = m._states["CAM-A"]
        self.assertFalse(st.tamper.active, "a noisy first frame must not poison the baseline")
        self.assertEqual(st.tamper.raises, 0)

    def test_first_frame_flat_does_not_become_baseline(self):
        m = manager(features=[FEATURE_TAMPER])
        flat = np.full((M.SAMPLE_HEIGHT, M.SAMPLE_WIDTH), 128, np.uint8)
        m.process_sample("CAM-A", flat, 0.0, {FEATURE_TAMPER})
        feed(m, "CAM-A", scene(62), 1.0, 40, features={FEATURE_TAMPER})
        st = m._states["CAM-A"]
        self.assertFalse(st.tamper.active)
        self.assertEqual(st.tamper.raises, 0)

    def test_partial_frame_is_rejected_as_unstructured(self):
        m = manager(features=[FEATURE_TAMPER])
        good = scene(63)
        black = np.zeros((M.SAMPLE_HEIGHT, M.SAMPLE_WIDTH), np.uint8)
        for i, img in enumerate([black, black, black]):
            m.process_sample("CAM-A", img, float(i), {FEATURE_TAMPER})
        st = m._states["CAM-A"]
        self.assertFalse(st.scene_baseline_valid())
        self.assertGreater(st.scene_rejected, 0)
        feed(m, "CAM-A", good, 10.0, 40, features={FEATURE_TAMPER})
        self.assertTrue(st.scene_baseline_valid())
        self.assertFalse(st.tamper.active)

    def test_stable_frames_make_the_baseline_valid(self):
        m = manager(features=[FEATURE_TAMPER])
        feed(m, "CAM-A", scene(64), 0.0, D.SCENE_WARMUP_SAMPLES + D.SCENE_BASELINE_SAMPLES,
             features={FEATURE_TAMPER})
        st = m._states["CAM-A"]
        self.assertTrue(st.scene_baseline_valid())
        self.assertEqual(st.scene_phase, "ready")

    def test_no_tamper_decision_before_baseline_is_valid(self):
        m = manager(features=[FEATURE_TAMPER])
        alt = other_scene(65)
        feed(m, "CAM-A", alt, 0.0, 3, features={FEATURE_TAMPER})
        st = m._states["CAM-A"]
        self.assertFalse(st.scene_baseline_valid())
        self.assertEqual(st.tamper.raises, 0)

    def test_normal_scene_after_baseline_gives_no_tamper(self):
        m = manager(features=[FEATURE_TAMPER])
        t = prime(m, "CAM-A", scene(66), 0.0, features={FEATURE_TAMPER})
        feed(m, "CAM-A", scene(66), t, 60, features={FEATURE_TAMPER})
        self.assertFalse(m._states["CAM-A"].tamper.active)

    def test_poisoned_start_then_normal_scene_recovers_with_no_tamper(self):
        """The exact live failure: bad first frame, then a normal reception."""
        m = manager(features=[FEATURE_TAMPER])
        flat = np.full((M.SAMPLE_HEIGHT, M.SAMPLE_WIDTH), 128, np.uint8)
        noise = RNG.integers(0, 255, (M.SAMPLE_HEIGHT, M.SAMPLE_WIDTH), dtype=np.uint8)
        for i, img in enumerate([flat, noise, flat]):
            m.process_sample("CAM-A", img, float(i), {FEATURE_TAMPER})
        feed(m, "CAM-A", scene(67), 5.0, 60, features={FEATURE_TAMPER})
        st = m._states["CAM-A"]
        self.assertTrue(st.scene_baseline_valid(), "must build a valid baseline once frames are good")
        self.assertEqual(st.tamper.raises, 0, "must not raise tamper at all")

    def test_genuine_tamper_after_valid_baseline_is_still_detected(self):
        m = manager(features=[FEATURE_TAMPER])
        base = scene(68)
        t = prime(m, "CAM-A", base, 0.0, features={FEATURE_TAMPER})
        feed(m, "CAM-A", np.roll(base, 160, axis=1), t, 20, features={FEATURE_TAMPER})
        st = m._states["CAM-A"]
        self.assertTrue(st.tamper.active)
        self.assertEqual(st.tamper.raises, 1)

    def test_genuine_tamper_does_not_adapt_the_baseline(self):
        m = manager(features=[FEATURE_TAMPER])
        base = scene(69)
        t = prime(m, "CAM-A", base, 0.0, features={FEATURE_TAMPER})
        before = m._states["CAM-A"].scene_baseline.copy()
        feed(m, "CAM-A", other_scene(70), t, 120, features={FEATURE_TAMPER})
        st = m._states["CAM-A"]
        self.assertTrue(st.tamper.active)
        self.assertTrue(np.allclose(before, st.scene_baseline, atol=1e-6),
                        "baseline must stay frozen while a real tamper is active")
        self.assertEqual(st.scene_rebuilds, 0, "and must not re-baseline this soon")

    def test_long_running_incident_is_not_recovered_by_a_rebaseline(self):
        """
        CHANGED 2026-09-19 (was test_stale_active_baseline_is_re_established_once).

        The old behaviour forced a re-baseline after 900 s active and cleared
        the condition in the same step - a Camera Tampering RECOVERY for a view
        that never recovered (R25 lights-off, 2026-09-17 21:19). A long-running
        incident must now stay open, keep its pre-incident baseline, and clear
        only when the view actually returns to it.
        """
        m = manager(features=[FEATURE_TAMPER])
        base = scene(71)
        t = prime(m, "CAM-A", base, 0.0, features={FEATURE_TAMPER})
        st = m._states["CAM-A"]
        before = st.scene_baseline.copy()
        changed = other_scene(72)
        seen = feed(m, "CAM-A", changed, t, int(D.SCENE_REBASELINE_AFTER_SECONDS) + 70,
                    features={FEATURE_TAMPER})
        self.assertEqual([a for _, f, a in seen], ["raise"], "one raise, no forced recovery")
        self.assertTrue(st.tamper.active, "still active after 900 s of a changed view")
        self.assertEqual(st.scene_rebuilds, 0, "no forced re-baseline")
        self.assertTrue(np.allclose(before, st.scene_baseline, atol=1e-6),
                        "the pre-incident baseline is preserved")
        t2 = t + int(D.SCENE_REBASELINE_AFTER_SECONDS) + 70
        seen = feed(m, "CAM-A", base, t2, int(D.TAMPER_SECONDS) + 3, features={FEATURE_TAMPER})
        self.assertEqual([a for _, f, a in seen], ["recover"],
                         "recovers only when the view returns to the pre-incident baseline")

    def test_recovery_from_genuine_tamper(self):
        m = manager(features=[FEATURE_TAMPER])
        base = scene(73)
        t = prime(m, "CAM-A", base, 0.0, features={FEATURE_TAMPER})
        feed(m, "CAM-A", other_scene(74), t, 20, features={FEATURE_TAMPER})
        st = m._states["CAM-A"]
        self.assertTrue(st.tamper.active)
        # A recovery waits the same window as the raise (see
        # test_camera_health_recovery.py), so a few good samples are no longer
        # enough to clear it.
        feed(m, "CAM-A", base, t + 20, int(D.TAMPER_SECONDS) + 3,
             features={FEATURE_TAMPER})
        self.assertFalse(st.tamper.active)
        self.assertEqual(st.tamper.recoveries, 1)

    def test_lighting_change_still_not_tamper(self):
        m = manager(features=[FEATURE_TAMPER])
        t = prime(m, "CAM-A", scene(75), 0.0, features={FEATURE_TAMPER})
        feed(m, "CAM-A", dim(scene(75), 0.55), t, 40, features={FEATURE_TAMPER})
        self.assertFalse(m._states["CAM-A"].tamper.active)

    def test_people_walking_still_not_tamper(self):
        m = manager(features=[FEATURE_TAMPER])
        base = scene(76)
        def with_person(i):
            img = base.copy()
            x = 10 + (i * 9) % (M.SAMPLE_WIDTH - 30)
            img[70:150, x:x + 22] = 30
            return img
        t = prime(m, "CAM-A", with_person, 0.0, features={FEATURE_TAMPER})
        feed(m, "CAM-A", with_person, t, 60, features={FEATURE_TAMPER})
        self.assertFalse(m._states["CAM-A"].tamper.active)

    def test_two_cameras_keep_independent_baselines(self):
        m = manager(features=[FEATURE_TAMPER], cameras=["CAM-A", "CAM-B"])
        flat = np.full((M.SAMPLE_HEIGHT, M.SAMPLE_WIDTH), 128, np.uint8)
        m.process_sample("CAM-A", flat, 0.0, {FEATURE_TAMPER})       # poison A's first frame
        ta = prime(m, "CAM-A", scene(77), 1.0, features={FEATURE_TAMPER})
        tb = prime(m, "CAM-B", scene(78), 0.0, features={FEATURE_TAMPER})
        a, b = m._states["CAM-A"], m._states["CAM-B"]
        self.assertTrue(a.scene_baseline_valid() and b.scene_baseline_valid())
        self.assertFalse(np.allclose(a.scene_baseline, b.scene_baseline, atol=1e-3))
        feed(m, "CAM-A", other_scene(79), ta, 20, features={FEATURE_TAMPER})
        feed(m, "CAM-B", scene(78), tb, 20, features={FEATURE_TAMPER})
        self.assertTrue(a.tamper.active)
        self.assertFalse(b.tamper.active, "one camera's tamper must not touch another's")
        self.assertEqual(b.scene_rejected, 0, "and A's bad frame must not count against B")



if __name__ == "__main__":
    unittest.main(verbosity=2)
