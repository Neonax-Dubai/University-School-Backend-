import unittest

import helpers  # noqa: F401
from helpers import Track

import phone_use as P


class PhoneUseTests(unittest.TestCase):
    def setUp(self):
        self.d = P.PhoneUseDetector()
        self.d.set_enabled_cameras(["camera_01"])
        self.student = Track("P-1", [100, 100, 300, 500])
        self.phone_in_hand = (0.6, [180, 250, 220, 290])          # centre (200, 270): upper body

    def feed(self, seconds, fps=2.0, phone=None, cam="camera_01", start=1000.0, tracks=None):
        out = []
        for i in range(int(seconds * fps) + 1):
            out += self.d.update(cam, tracks or [self.student], [phone or self.phone_in_hand], start + i / fps)
        return out

    def test_attach_prefers_the_smallest_containing_person_in_the_upper_region(self):
        big, small = [0, 0, 1000, 1000], [150, 200, 260, 480]
        self.assertEqual(P.attach([190, 250, 210, 270], [big, small]), 1)
        self.assertIsNone(P.attach([190, 470, 210, 478], [small]), "near the feet = not hand/face region")
        self.assertIsNone(P.attach([500, 500, 520, 520], [small]))

    def test_sustained_phone_raises_once_with_evidence_box(self):
        found = self.feed(8)
        self.assertEqual(len(found), 1)
        f = found[0]
        self.assertEqual(f.track_id, "P-1")
        self.assertEqual(f.bbox, [100, 100, 300, 500])
        self.assertGreaterEqual(f.metadata()["duration_seconds"], P.PHONE_MIN_SECONDS)
        self.assertEqual(f.scope(), "phone:P-1")

    def test_cooldown_stops_a_repeat(self):
        self.assertEqual(len(self.feed(8)), 1)
        self.assertEqual(self.feed(20, start=1010.0), [], "inside the cooldown")
        self.assertEqual(len(self.feed(8, start=1000.0 + P.PHONE_COOLDOWN_SECONDS + 20)), 1)

    def test_a_brief_glimpse_does_not_raise(self):
        self.assertEqual(self.feed(2), [])

    def test_low_confidence_boxes_are_ignored(self):
        self.assertEqual(self.feed(10, phone=(0.1, [180, 250, 220, 290])), [])

    def test_disarmed_camera_does_nothing(self):
        self.assertEqual(self.feed(10, cam="camera_02"), [])

    def test_forget_camera_clears_state(self):
        self.feed(3)
        self.d.forget_camera("camera_01")
        self.assertEqual(self.d.stats()["tracked"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
