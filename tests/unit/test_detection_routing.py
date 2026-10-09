"""
Where a detection goes (detection_routing.route): the mobile-phone analytic keeps exactly its old
input; a phone ALSO reaches the tracker only when "cell phone" is an unattended class on that camera.
"""
import unittest
from unittest import mock

import helpers  # noqa: F401

import abandoned
import detection_routing as R
import phone_use

PERSON, BACKPACK, LAPTOP, PHONE, BOOK = 0, 24, 63, 67, 73
ALL = {PERSON, BACKPACK, LAPTOP, PHONE}


def route(class_id, confidence, allowed=ALL, abandon=True, phone=True, floor=0.45):
    return R.route(class_id, confidence, allowed=allowed, min_confidence=floor, abandon_armed=abandon,
                   phone_armed=phone)


def old_route(class_id, confidence, allowed=ALL, abandon=True, phone=True, floor=0.45, candidates=(24, 26, 28)):
    """zayed_inference.process_result's loop as deployed in f4e16a9, verbatim in effect."""
    if class_id not in allowed:
        return False, False
    if phone and class_id == phone_use.PHONE_CLASS_ID and confidence >= phone_use.PHONE_CONFIDENCE:
        return True, False
    if confidence < floor and not (abandon and class_id in candidates and confidence >= abandoned.OBJECT_CONFIDENCE):
        return False, False
    return False, True


class PhoneAnalyticTests(unittest.TestCase):
    def test_the_phone_analytic_receives_exactly_what_it_did_before(self):
        for abandon in (True, False):
            for phone in (True, False):
                for confidence in (0.10, 0.25, 0.29, 0.30, 0.31, 0.45, 0.90):
                    self.assertEqual(route(PHONE, confidence, abandon=abandon, phone=phone)[0],
                                     old_route(PHONE, confidence, abandon=abandon, phone=phone)[0],
                                     (abandon, phone, confidence))

    def test_a_phone_reaches_the_tracker_too_when_cell_phone_is_an_unattended_class(self):
        self.assertEqual(route(PHONE, 0.6), (True, True))
        self.assertEqual(route(PHONE, 0.27), (False, True), "below the phone floor, above the unattended floor")
        self.assertEqual(route(PHONE, 0.20), (False, False))

    def test_without_cell_phone_as_an_unattended_class_phones_stay_out_of_the_tracker(self):
        with mock.patch.object(abandoned, "CANDIDATE_CLASS_IDS", frozenset({24, 26, 28})):
            self.assertEqual(route(PHONE, 0.6), (True, False))
            self.assertEqual(route(PHONE, 0.6), old_route(PHONE, 0.6))
        self.assertEqual(route(PHONE, 0.6, abandon=False), (True, False), "unattended not armed on the camera")

    def test_with_phone_detection_off_a_phone_is_only_an_unattended_candidate(self):
        self.assertEqual(route(PHONE, 0.6, phone=False), (False, True))
        self.assertEqual(route(PHONE, 0.6, phone=False, abandon=False), (False, True), "camera floor 0.45 met")
        self.assertEqual(route(PHONE, 0.40, phone=False, abandon=False), (False, False))


class OtherClassesTests(unittest.TestCase):
    def test_everything_but_phones_routes_as_before(self):
        for class_id in (PERSON, BACKPACK, BOOK):
            for abandon in (True, False):
                for confidence in (0.2, 0.25, 0.3, 0.45, 0.8):
                    allowed = ALL | {BOOK}
                    self.assertEqual(route(class_id, confidence, allowed, abandon),
                                     old_route(class_id, confidence, allowed, abandon), (class_id, abandon, confidence))

    def test_a_laptop_is_tracked_at_the_unattended_floor_only_when_unattended_is_armed(self):
        self.assertEqual(route(LAPTOP, 0.27), (False, True))
        self.assertEqual(route(LAPTOP, 0.27, abandon=False), (False, False))
        self.assertEqual(route(LAPTOP, 0.6, allowed={PERSON}), (False, False), "not a class this camera needs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
