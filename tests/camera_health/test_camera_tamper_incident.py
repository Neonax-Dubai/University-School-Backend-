"""
Camera Tampering incident semantics (2026-09-19). CPU only, no camera, no network.

    aienv/bin/python test_camera_tamper_incident.py

Camera Tampering = deliberate physical interference with the view (lens
covered, object over the lens). Only the scene-change and obstruction detectors
feed it; defocus, signal loss and reconnects must not. Evidence and numbers:
/home/matrix/Dubai_Police/camera_tamper_R25_audit_2026-09-18/
camera_tamper_targeted_validation_2026-09-18.md

The P3 / P4 / N1 tests replay the REAL per-second scene-distance series measured
on CAM-R25's recordings through the production state machine (Condition), with
the production decision rule (distance >= TAMPER_DISTANCE).
"""
import os
import sys
import unittest

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from camera_health import detectors as D                              # noqa: E402
from camera_health import metrics as M                                # noqa: E402
from camera_health.manager import (CameraHealthManager, CAMERA_HEALTH_FEATURES,  # noqa: E402
                                   CAMERA_TAMPERING_FEATURES, FEATURE_DEFOCUS,
                                   FEATURE_OBSTRUCTION, FEATURE_SIGNAL_LOSS,
                                   FEATURE_TAMPER, UMBRELLA_EVENT_TYPE)
from camera_health.state import Condition, PHASE_READY, PHASE_STARTING  # noqa: E402
from test_camera_health import (FakeReader, blur, covered_dark, other_scene,  # noqa: E402
                                scene)

CAM = "CAM-R25"
IMAGE_FEATURES = {FEATURE_OBSTRUCTION, FEATURE_DEFOCUS, FEATURE_TAMPER}
PERSIST = 8.0      # validated reference; asserted against the code default below

# --- real CAM-R25 scene-distance series, 1 sample/s (offline replay 2026-09-18) ---
# P3 17 Sep 12:19:15..12:19:50 - box waved over the lens (the dip at 12:19:35 is real)
P3 = [0.0361, 0.0343, 0.0328, 0.0307, 0.0286, 0.059, 0.3715, 0.5835, 0.5375, 1.0179,
      1.0518, 0.8815, 0.8827, 0.7591, 0.8431, 1.093, 0.7979, 0.7452, 0.9478, 0.8917,
      0.6356, 0.6223, 0.7329, 0.9432, 0.963, 0.7618, 0.1487, 0.092, 0.082, 0.0848,
      0.0833, 0.0795, 0.0742, 0.0736, 0.0688, 0.0719]
# P4 16 Sep 16:25:52..16:26:22 - keyboard box pressed over the lens
P4 = [0.0092, 0.0214, 0.0327, 0.0477, 0.1477, 0.3285, 0.6097, 0.6799, 1.0076, 1.3483,
      1.3662, 1.378, 1.3758, 1.3595, 1.3699, 1.3914, 1.3693, 1.398, 1.3841, 0.0775,
      0.065, 0.0654, 0.0622, 0.0895, 0.243, 0.3126, 0.3828, 0.3239, 0.1449, 0.1395, 0.1318]
# N1 17 Sep 12:12:20..12:12:45 - ~8 people walking under the camera (5 s fired on this)
N1 = [0.3838, 0.5147, 0.628, 0.6852, 0.7217, 0.7346, 0.6647, 0.7346, 0.779, 0.5398,
      0.413, 0.2475, 0.1785, 0.153, 0.1357, 0.1157, 0.1172, 0.1261, 0.1934, 0.3065,
      0.4889, 0.5326, 0.2144, 0.2108, 0.1963, 0.1821]
QUIET = [0.03] * 20


def run_series(series, start=0.0, condition=None, persist=PERSIST):
    """Drive the production Condition with a distance series; return [(t, action)]."""
    c = condition or Condition("camera_tamper")
    out = []
    for i, d in enumerate(series):
        t = start + i
        a = c.update(d >= D.TAMPER_DISTANCE, t, persist, D.TAMPER_COOLDOWN_SECONDS, {"distance": d})
        if a:
            out.append((t, a))
    return out, c


class Harness:
    """A live-mode manager with an in-memory event sink - nothing leaves the process."""

    def __init__(self, features=IMAGE_FEATURES):
        self.sent = []
        self.log = []
        self.m = CameraHealthManager(shadow=False, logger=self.log.append,
                                     event_sink=lambda **kw: self.sent.append(kw))
        self.m.set_enabled_cameras({CAM: {f: True for f in CAMERA_HEALTH_FEATURES}})
        self.features = set(features)
        self.t = 0.0

    def feed(self, img, count):
        for _ in range(count):
            self.m.process_sample(CAM, img, self.t, self.features)
            self.t += 1.0

    def observe_frame(self, frame, count):
        """Production path: observe() (frame size, to_sample) then the worker's _process()."""
        for _ in range(count):
            self.m.observe(CAM, frame, self.t)
            with self.m._lock:
                item = self.m._pending.pop(CAM, None)
            if item is not None:
                gray, ts = item
                self.m._process(self.m._states[CAM], gray, ts, self.features)
            self.t += 1.0

    def prime(self, img=None, count=D.BASELINE_SAMPLES + 5):
        self.feed(scene(3) if img is None else img, count)

    def ct(self):
        return [(e["action"], e["metadata"].get("reason"), e["event_type"]) for e in self.sent]

    @property
    def state(self):
        return self.m._states[CAM]


def full_frame(gray320, w, h):
    return cv2.cvtColor(cv2.resize(gray320, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)


class ConfigTests(unittest.TestCase):
    def test_validated_defaults(self):
        if "CAMERA_HEALTH_TAMPER_SECONDS" not in os.environ:
            self.assertEqual(D.TAMPER_SECONDS, 8.0)
        if "CAMERA_HEALTH_TAMPER_COOLDOWN_SECONDS" not in os.environ:
            self.assertEqual(D.TAMPER_COOLDOWN_SECONDS, 0.0)
        self.assertEqual(D.OBSTRUCTION_COOLDOWN_SECONDS, 0.0)
        self.assertEqual(D.TAMPER_DISTANCE, 0.65, "threshold unchanged")
        self.assertEqual(set(CAMERA_TAMPERING_FEATURES), {FEATURE_TAMPER, FEATURE_OBSTRUCTION})


class RoutingTests(unittest.TestCase):
    def test_01_scene_change_is_camera_tampering(self):
        h = Harness(); h.prime(); h.feed(other_scene(9), 12)
        self.assertEqual(h.ct(), [("raise", FEATURE_TAMPER, UMBRELLA_EVENT_TYPE)])

    def test_02_obstruction_is_camera_tampering(self):
        h = Harness(); h.prime(); h.feed(covered_dark(scene(3)), 12)
        self.assertEqual(h.ct(), [("raise", FEATURE_OBSTRUCTION, UMBRELLA_EVENT_TYPE)])

    def test_03_defocus_is_not_camera_tampering(self):
        h = Harness(); h.prime(); h.feed(blur(scene(3), 25), 60)
        self.assertTrue(h.state.defocus.active, "defocus is still detected (diagnostic)")
        self.assertEqual(h.sent, [])
        self.assertTrue(any("diagnostic only" in l for l in h.log))

    def test_04_signal_loss_is_not_camera_tampering(self):
        h = Harness()
        reader = FakeReader(100)
        for t in range(30):
            h.m.note_streams({CAM: reader}, now=float(t))          # counter stuck
        self.assertTrue(h.m._states[CAM].signal.active, "signal loss is still detected")
        reader.frames_read = 200
        h.m.note_streams({CAM: reader}, now=31.0)
        self.assertFalse(h.m._states[CAM].signal.active)
        self.assertEqual(h.sent, [])

    def test_05_reconnect_is_not_camera_tampering(self):
        """Counter stalls (reconnect), frames resume on the same view, same and new size."""
        h = Harness()
        reader = FakeReader(0)
        base = scene(3)
        for _ in range(40):
            reader.frames_read += 25
            h.m.note_streams({CAM: reader}, now=h.t)
            h.observe_frame(full_frame(base, 640, 360), 1)
        for _ in range(8):                                          # 8 s outage
            h.m.note_streams({CAM: reader}, now=h.t); h.t += 1.0
        for size in ((640, 360), (800, 600)):                       # same profile, then a new one
            for _ in range(60):
                reader.frames_read += 25
                h.m.note_streams({CAM: reader}, now=h.t)
                h.observe_frame(full_frame(base, *size), 1)
        self.assertEqual(h.sent, [])


class PersistenceTests(unittest.TestCase):
    def _changed_for(self, seconds):
        h = Harness(); h.prime()
        h.feed(other_scene(9), seconds); h.feed(scene(3), 20)
        return h

    def test_06_people_like_change_5s_no_alert(self):
        self.assertEqual(self._changed_for(5).sent, [])

    def test_07_people_like_change_6s_no_alert(self):
        self.assertEqual(self._changed_for(6).sent, [])

    def test_08_genuine_11s_interference_alerts(self):
        h = self._changed_for(11)
        self.assertEqual([a for a, _, _ in h.ct()], ["raise", "recover"])

    def test_n1_real_people_series_no_alert_at_8s(self):
        self.assertEqual(run_series(QUIET + N1 + QUIET)[0], [])

    def test_n1_real_people_series_did_alert_at_5s(self):
        """Control: the same series DOES fire at the old 5 s - the test is meaningful."""
        self.assertEqual([a for _, a in run_series(QUIET + N1 + QUIET, persist=5.0)[0]],
                         ["raise", "recover"])

    def test_09_p3_real_series_detected_at_8s(self):
        acts, _ = run_series(QUIET + P3 + QUIET)
        self.assertEqual([a for _, a in acts], ["raise", "recover"])
        self.assertEqual(acts[0][0], len(QUIET) + 17, "confirmed at 12:19:32, as in the replay")

    def test_10_p4_real_series_detected_at_8s(self):
        acts, _ = run_series(QUIET + P4 + QUIET)
        self.assertEqual([a for _, a in acts], ["raise", "recover"])


class CooldownTests(unittest.TestCase):
    def test_11_p3_not_suppressed_by_previous_incident(self):
        """12:12 incident, cleared, then P3 seven minutes later -> a NEW alert."""
        acts, c = run_series(QUIET + [1.0] * 12 + QUIET)
        acts2, _ = run_series(QUIET * 20 + P3 + QUIET, start=len(QUIET) * 2 + 12, condition=c)
        self.assertEqual([a for _, a in acts], ["raise", "recover"])
        self.assertEqual([a for _, a in acts2], ["raise", "recover"])

    def test_12_p4_not_suppressed_by_previous_incident(self):
        """P2 at 16:22 then P4 at 16:26 (was suppressed by the 600 s cooldown)."""
        acts, c = run_series(QUIET + [1.3] * 12 + QUIET)
        gap = [0.03] * 200
        acts2, _ = run_series(gap + P4 + QUIET, start=len(QUIET) * 2 + 12, condition=c)
        self.assertEqual([a for _, a in acts2], ["raise", "recover"])


class LifecycleTests(unittest.TestCase):
    def test_13_14_15_16_one_alert_no_duplicates_one_recovery_new_alert(self):
        h = Harness(); h.prime()
        h.feed(other_scene(9), 90)                       # tamper continues 90 s
        self.assertEqual(h.ct(), [("raise", FEATURE_TAMPER, UMBRELLA_EVENT_TYPE)],
                         "one alert, no duplicates while it continues")
        h.feed(scene(3), 20)                             # view back to normal
        self.assertEqual([a for a, _, _ in h.ct()], ["raise", "recover"], "one recovery")
        first_id = h.sent[0]["metadata"]["incident_id"]
        self.assertEqual(h.sent[1]["metadata"]["incident_id"], first_id,
                         "the recovery carries its own incident's id")
        h.feed(other_scene(9), 12)                       # a later, new tamper
        self.assertEqual([a for a, _, _ in h.ct()], ["raise", "recover", "raise"], "new alert")
        self.assertNotEqual(h.sent[2]["metadata"]["incident_id"], first_id)

    def test_new_obstruction_incident_after_recovery(self):
        h = Harness(); h.prime()
        for _ in range(2):
            h.feed(covered_dark(scene(3)), 12); h.feed(scene(3), 12)
        self.assertEqual([a for a, _, _ in h.ct()], ["raise", "recover", "raise", "recover"])

    def test_17_defocus_recovery_does_not_close_tamper(self):
        h = Harness()
        h.m._announce(CAM, FEATURE_TAMPER, "raise", {"distance": 1.2}, 100.0)
        h.m._announce(CAM, FEATURE_DEFOCUS, "raise", {"ratio": 0.3}, 101.0)
        h.m._announce(CAM, FEATURE_DEFOCUS, "recover", {"ratio": 0.9}, 110.0)
        self.assertEqual([a for a, _, _ in h.ct()], ["raise"])
        h.m._announce(CAM, FEATURE_TAMPER, "recover", {"distance": 0.05}, 120.0)
        self.assertEqual([a for a, _, _ in h.ct()], ["raise", "recover"])

    def test_18_signal_loss_recovery_does_not_close_tamper(self):
        h = Harness()
        h.m._announce(CAM, FEATURE_TAMPER, "raise", {"distance": 1.2}, 100.0)
        h.m._announce(CAM, FEATURE_SIGNAL_LOSS, "raise", {"gap_seconds": 5.1}, 101.0)
        h.m._announce(CAM, FEATURE_SIGNAL_LOSS, "recover", {"gap_seconds": 0.0}, 110.0)
        self.assertEqual([a for a, _, _ in h.ct()], ["raise"])

    def test_19_forced_rebaseline_emits_no_recovery(self):
        h = Harness(features={FEATURE_TAMPER}); h.prime()
        h.feed(other_scene(9), int(D.SCENE_REBASELINE_AFTER_SECONDS) + 60)
        self.assertEqual([a for a, _, _ in h.ct()], ["raise"])
        self.assertTrue(h.state.tamper.active)
        self.assertEqual(h.state.scene_rebuilds, 0)


class FrameSizeTests(unittest.TestCase):
    def test_20_frame_size_change_resets_and_relearns(self):
        h = Harness(); base = scene(3)
        h.observe_frame(full_frame(base, 640, 360), 40)
        st = h.state
        self.assertEqual(st.scene_phase, PHASE_READY)
        self.assertIsNotNone(st.sharpness_baseline)
        h.observe_frame(full_frame(base, 800, 600), 1)
        self.assertEqual(st.profile_resets, 1)
        self.assertIsNone(st.sharpness_baseline, "sharpness baseline invalidated")
        self.assertIsNone(st.edge_baseline, "edge baseline invalidated")
        self.assertIsNone(st.scene_baseline, "scene baseline invalidated")
        self.assertEqual(st.scene_phase, PHASE_STARTING, "first new-profile sample is warm-up, not judged")
        h.observe_frame(full_frame(base, 800, 600), 40)
        self.assertEqual(st.scene_phase, PHASE_READY)
        self.assertIsNotNone(st.sharpness_baseline)
        self.assertTrue(any("frame size changed" in l for l in h.log))

    def _profile_change(self, reset):
        h = Harness(); base = scene(3)
        h.observe_frame(full_frame(base, 640, 360), 40)
        softer = blur(base, 9)       # the new profile's normal reads far less "sharp"
        h.m.observe(CAM, full_frame(softer, 800, 600), h.t)
        if not reset:
            h.state.profile_reset_pending = None       # emulate the old behaviour (control)
        with h.m._lock:
            gray, ts = h.m._pending.pop(CAM)
        h.m._process(h.state, gray, ts, h.features); h.t += 1
        h.observe_frame(full_frame(softer, 800, 600), 120)
        return h, softer

    def test_21_resolution_change_no_false_defocus(self):
        h, _ = self._profile_change(reset=True)
        self.assertFalse(h.state.defocus.active)
        self.assertEqual(h.sent, [])
        control, _ = self._profile_change(reset=False)
        self.assertTrue(control.state.defocus.active,
                        "control: WITHOUT the reset the carried baseline raises defocus")

    def test_22_genuine_tamper_after_resolution_change_detected(self):
        h, softer = self._profile_change(reset=True)
        h.observe_frame(full_frame(other_scene(9), 800, 600), 12)
        self.assertEqual(h.ct(), [("raise", FEATURE_TAMPER, UMBRELLA_EVENT_TYPE)])


class ObstructionBaselineTests(unittest.TestCase):
    def test_23_obstruction_does_not_contaminate_scene_baseline(self):
        h = Harness(); h.prime()
        before = h.state.scene_baseline.copy()
        h.feed(covered_dark(scene(3)), 180)
        self.assertTrue(h.state.obstruction.active)
        self.assertTrue(np.allclose(before, h.state.scene_baseline, atol=1e-6))

    def test_24_recovery_after_obstruction_no_false_tamper(self):
        h = Harness(); h.prime()
        h.feed(covered_dark(scene(3)), 180)
        h.feed(scene(3), 60)
        self.assertEqual(h.ct(), [("raise", FEATURE_OBSTRUCTION, UMBRELLA_EVENT_TYPE),
                                  ("recover", FEATURE_OBSTRUCTION, UMBRELLA_EVENT_TYPE)])
        self.assertFalse(h.state.tamper.active)


class ReasonTests(unittest.TestCase):
    def test_25_p1_reason_is_the_tamper_detector_not_defocus(self):
        """A cover that is BOTH a scene change and blurred: defocus confirms first
        (5 s) but must not become the Camera Tampering reason."""
        h = Harness(); h.prime()
        cover = blur(other_scene(9), 31)
        h.feed(cover, 12)
        self.assertTrue(h.state.defocus.active, "defocus did confirm (and first)")
        self.assertEqual(h.ct(), [("raise", FEATURE_TAMPER, UMBRELLA_EVENT_TYPE)])
        meta = h.sent[0]["metadata"]
        self.assertNotIn(FEATURE_DEFOCUS, meta["reasons"])
        self.assertEqual(meta["reason_label"], "Scene changed")
        self.assertEqual(meta["semantic_note"] if "semantic_note" in meta else
                         h.sent[0]["metadata"].get("semantic_note"), "scene-change heuristic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
