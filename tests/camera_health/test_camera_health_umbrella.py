"""
Camera Tampering: ONE incident per interference, from the tampering detectors only.

    aienv/bin/python test_camera_health_umbrella.py

THE ORIGINAL DEFECT. A lens covered on CAM-R25 on 2026-09-16 at 16:22 produced
camera_tamper at 16:22:02, camera_defocus at 16:22:04, and a separate recovery
for each: four events and two alarms for one person holding one object over one
camera. Detectors that see the same interference are reported ONCE.

CHANGED 2026-09-19. Only scene change (camera_tamper) and obstruction are
Camera Tampering (manager.CAMERA_TAMPERING_FEATURES). Defocus and signal loss
used to join, open and close the incident; they are now diagnostics only. The
tests that used defocus / signal loss as the "second detector" use obstruction
instead, and the ones that asserted signal loss / defocus BECOME Camera
Tampering now assert that they do not. Originals: *.pre-tamper-routing.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import camera_health                                                   # noqa: E402
from camera_health.manager import (FEATURE_DEFOCUS, FEATURE_OBSTRUCTION,  # noqa: E402
                                   FEATURE_SIGNAL_LOSS, FEATURE_TAMPER,
                                   UMBRELLA_EVENT_TYPE)

CAM = "CAM-R25"


class UmbrellaIncidentTests(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.manager = camera_health.CameraHealthManager(
            shadow=False, logger=lambda line: None,
            event_sink=lambda **kw: self.sent.append(dict(kw)))

    def announce(self, feature, action, at, **detail):
        """One detector transition, as _process() would report it."""
        self.manager._announce(CAM, feature, action, detail or {"x": 1}, at)

    def types(self):
        return [(e["event_type"], e["metadata"].get("reason"),
                 bool(e["metadata"].get("recovered"))) for e in self.sent]

    # ------------------------------------------------------------- the bug
    def test_two_detectors_one_incident_one_event(self):
        self.announce(FEATURE_TAMPER, "raise", 100.0, distance=1.28)
        self.announce(FEATURE_OBSTRUCTION, "raise", 102.0, edge_density=0.01)
        self.assertEqual(len(self.sent), 1, "a second detector must not open a second alarm")
        self.assertEqual(self.sent[0]["event_type"], UMBRELLA_EVENT_TYPE)

    def test_the_incident_ends_only_when_every_condition_clears(self):
        self.announce(FEATURE_TAMPER, "raise", 100.0)
        self.announce(FEATURE_OBSTRUCTION, "raise", 102.0)
        self.announce(FEATURE_OBSTRUCTION, "recover", 113.0)
        self.assertEqual(len(self.sent), 1, "tamper is still active")
        self.announce(FEATURE_TAMPER, "recover", 115.0)
        self.assertEqual(len(self.sent), 2)
        self.assertTrue(self.sent[-1]["metadata"]["recovered"])

    def test_signal_loss_is_not_camera_tampering(self):
        # was test_everything_is_the_one_event_type (signal loss -> camera_tamper)
        self.announce(FEATURE_SIGNAL_LOSS, "raise", 100.0, gap_seconds=5.1)
        self.announce(FEATURE_SIGNAL_LOSS, "recover", 120.0, gap_seconds=0.0)
        self.assertEqual(self.sent, [])

    # ------------------------------------------------------------- reasons
    def test_the_reason_says_which_detector_noticed(self):
        # was: signal loss as the reason. Obstruction is the other routed detector.
        self.announce(FEATURE_OBSTRUCTION, "raise", 100.0, edge_density=0.01)
        meta = self.sent[0]["metadata"]
        self.assertEqual(meta["reason"], FEATURE_OBSTRUCTION)
        self.assertEqual(meta["reason_label"], "View obstructed")

    def test_every_reason_seen_is_listed(self):
        self.announce(FEATURE_TAMPER, "raise", 100.0)
        self.announce(FEATURE_OBSTRUCTION, "raise", 102.0)
        self.announce(FEATURE_OBSTRUCTION, "recover", 110.0)
        self.announce(FEATURE_TAMPER, "recover", 112.0)
        meta = self.sent[-1]["metadata"]
        self.assertEqual(meta["reasons"], [FEATURE_TAMPER, FEATURE_OBSTRUCTION])
        self.assertEqual(meta["reason_labels"], ["Scene changed", "View obstructed"])

    def test_the_detectors_own_numbers_survive(self):
        self.announce(FEATURE_TAMPER, "raise", 100.0, distance=1.28)
        self.announce(FEATURE_OBSTRUCTION, "raise", 102.0, edge_density=0.01)
        self.announce(FEATURE_TAMPER, "recover", 112.0, distance=0.06)
        self.announce(FEATURE_OBSTRUCTION, "recover", 114.0, edge_density=0.47)
        conditions = self.sent[-1]["metadata"]["conditions"]
        self.assertEqual(conditions[FEATURE_OBSTRUCTION]["edge_density"], 0.47)
        self.assertEqual(conditions[FEATURE_TAMPER]["distance"], 0.06)

    def test_recovery_carries_how_long_it_lasted(self):
        self.announce(FEATURE_TAMPER, "raise", 100.0)
        self.announce(FEATURE_TAMPER, "recover", 111.5)
        self.assertEqual(self.sent[-1]["metadata"]["duration_seconds"], 11.5)

    # ------------------------------------------------------------ sequence
    def test_a_later_incident_raises_again(self):
        self.announce(FEATURE_TAMPER, "raise", 100.0)
        self.announce(FEATURE_TAMPER, "recover", 110.0)
        self.announce(FEATURE_OBSTRUCTION, "raise", 900.0)
        self.assertEqual([t[2] for t in self.types()], [False, True, False])
        self.assertEqual(self.sent[-1]["metadata"]["reason"], FEATURE_OBSTRUCTION)

    def test_a_recovery_without_a_raise_emits_nothing(self):
        """A detector clearing on the first sample of a fresh process."""
        self.announce(FEATURE_TAMPER, "recover", 100.0)
        self.announce(FEATURE_DEFOCUS, "recover", 101.0)
        self.assertEqual(self.sent, [])

    def test_cameras_do_not_share_an_incident(self):
        self.manager._announce("CAM-A", FEATURE_TAMPER, "raise", {}, 100.0)
        self.manager._announce("CAM-B", FEATURE_TAMPER, "raise", {}, 101.0)
        self.assertEqual(len(self.sent), 2)
        self.assertEqual({e["camera_id"] for e in self.sent}, {"CAM-A", "CAM-B"})


if __name__ == "__main__":
    unittest.main(verbosity=1)
