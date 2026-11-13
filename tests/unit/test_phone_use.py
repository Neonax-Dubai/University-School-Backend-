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

    # ZAYED 2026-11-13: the cooldown also binds by seat, because track ids fragment.
    def test_same_seat_under_a_new_track_id_is_held_by_the_cooldown(self):
        self.assertEqual(len(self.feed(8)), 1)
        same_student_new_id = Track("P-2", [104, 98, 302, 506])
        self.assertEqual(self.feed(10, start=1060.0, tracks=[same_student_new_id]), [],
                         "a re-identified student at the same seat is not a new sighting")
        self.assertEqual(self.d.stats()["seat_suppressed"], 1)

    def test_a_different_seat_still_raises(self):
        self.assertEqual(len(self.feed(8)), 1)
        neighbour = Track("P-3", [600, 100, 800, 500])
        found = self.feed(10, start=1060.0, tracks=[neighbour], phone=(0.6, [680, 250, 720, 290]))
        self.assertEqual([f.track_id for f in found], ["P-3"])

    def test_the_same_seat_raises_again_after_the_cooldown(self):
        self.assertEqual(len(self.feed(8)), 1)
        later = Track("P-4", [100, 100, 300, 500])
        self.assertEqual(len(self.feed(8, start=1000.0 + P.PHONE_COOLDOWN_SECONDS + 20, tracks=[later])), 1)

    def test_seat_rule_can_be_switched_off(self):
        saved = P.PHONE_SEAT_IOU
        P.PHONE_SEAT_IOU = 0.0
        try:
            self.assertEqual(len(self.feed(8)), 1)
            self.assertEqual(len(self.feed(10, start=1060.0, tracks=[Track("P-2", [100, 100, 300, 500])])), 1)
        finally:
            P.PHONE_SEAT_IOU = saved

    def test_prune_forgets_old_seats(self):
        self.feed(8)
        self.d.prune(1000.0 + P.PHONE_COOLDOWN_SECONDS + 60)
        self.assertNotIn("camera_01", self.d._recent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
