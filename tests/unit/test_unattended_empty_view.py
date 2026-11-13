"""Unattended objects alert only once the room is empty (ZAYED 2026-11-13, owner decision).

    python -m pytest tests/unit/test_unattended_empty_view.py

WHY. In C102 the 30 s ownerless rule raised 178 "Unattended Object" alarms in two days, nearly all
laptops on the desks of an occupied class whose student had stepped away. The owner chose: an
unattended object is something LEFT BEHIND - it alerts only after its dwell AND once the camera
has seen nobody for EMPTY_VIEW_SECONDS. ABANDON_REQUIRE_EMPTY_VIEW=0 restores the old rule.
"""
import unittest

import helpers  # noqa: F401

import abandoned as A
from test_unattended_episodes import BAG, FAR_PERSON, NEAR_PERSON, Obj, Scene

LAPTOP = dict(class_id=63, class_name="laptop")


def laptop(t):
    return [Obj("O-1", BAG, **LAPTOP)]


class EmptyViewTests(unittest.TestCase):
    def setUp(self):
        self.saved = A.REQUIRE_EMPTY_VIEW
        A.REQUIRE_EMPTY_VIEW = True

    def tearDown(self):
        A.REQUIRE_EMPTY_VIEW = self.saved

    def test_a_laptop_left_during_class_never_alerts_while_people_are_in_view(self):
        s = Scene().run(600, objects=laptop, persons=lambda t: [FAR_PERSON])
        self.assertEqual(s.alerts, [], "students elsewhere in the room: stepped away, not left behind")

    def test_it_alerts_once_the_room_has_been_empty_long_enough(self):
        s = Scene().run(300, objects=laptop, persons=lambda t: [FAR_PERSON])
        left_at = s.t
        s.run(A.EMPTY_VIEW_SECONDS + 30, objects=laptop)
        self.assertEqual(len(s.alerts), 1)
        t, _cam, _track, meta = s.alerts[0]
        self.assertGreaterEqual(t - left_at, A.EMPTY_VIEW_SECONDS - 0.2)
        self.assertLess(t - left_at, A.EMPTY_VIEW_SECONDS + 1.0, "no extra delay once the dwell is done")
        self.assertGreaterEqual(meta["empty_view_seconds"], A.EMPTY_VIEW_SECONDS - 0.2)
        self.assertEqual(meta["episode_state"], "alerted")

    def test_the_owner_returning_before_the_room_empties_means_no_alert(self):
        s = Scene()
        s.run(120, objects=laptop, persons=lambda t: [FAR_PERSON])
        s.run(30, objects=laptop, persons=lambda t: [NEAR_PERSON, FAR_PERSON])     # owner back
        s.run(5, objects=lambda t: [], persons=lambda t: [NEAR_PERSON])            # takes it, leaves
        s.run(300, objects=lambda t: [], persons=lambda t: [])
        self.assertEqual(s.alerts, [])

    def test_left_in_an_already_empty_room_it_still_needs_its_dwell(self):
        s = Scene().run(A.EMPTY_VIEW_SECONDS + 60, objects=laptop)
        self.assertEqual(len(s.alerts), 1)
        self.assertGreaterEqual(s.alerts[0][3]["dwell_seconds"], 30.0)

    def test_someone_walking_through_restarts_the_empty_clock(self):
        s = Scene().run(120, objects=laptop, persons=lambda t: [FAR_PERSON])
        s.run(40, objects=laptop)
        s.run(2, objects=laptop, persons=lambda t: [FAR_PERSON])                  # cleaner passes
        resumed = s.t
        s.run(A.EMPTY_VIEW_SECONDS + 10, objects=laptop)
        self.assertEqual(len(s.alerts), 1)
        self.assertGreaterEqual(s.alerts[0][0] - resumed, A.EMPTY_VIEW_SECONDS - 0.2)

    def test_the_switch_restores_the_previous_rule(self):
        A.REQUIRE_EMPTY_VIEW = False
        s = Scene().run(60, objects=laptop, persons=lambda t: [FAR_PERSON])
        self.assertEqual(len(s.alerts), 1, "old rule: nobody NEAR is enough")
        self.assertNotIn("empty_view_seconds", s.alerts[0][3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
