import unittest
from datetime import datetime, timezone

import helpers  # noqa: F401
from helpers import Config, Track

import evacuation as E


def iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


class EvacuationTests(unittest.TestCase):
    def setUp(self):
        self.m = E.EvacuationMonitor("http://unused", "tok", log=lambda *_: None)
        self.m.configure([Config("camera_01", features={"evacuation_monitoring": True}),
                          Config("camera_02", features={"evacuation_monitoring": True}),
                          Config("camera_03", features={})])
        self.start = 10_000.0
        self.m.set_sessions({"sessions": [{"session_uid": "EVAC-1", "classroom_id": "C101", "status": "active",
                                           "started_at": iso(self.start), "evacuation_seconds": 180,
                                           "deadline_at": iso(self.start + 180)}]}, now=self.start)
        self.people = [Track("P-1", [10, 10, 50, 150]), Track("P-2", [200, 20, 260, 170])]

    def observe(self, t, cam="camera_01", people=None):
        return self.m.observe(cam, self.people if people is None else people, None, 2592, 1944, t)

    def test_nothing_before_the_deadline(self):
        for dt in range(0, 180, 5):
            self.assertEqual(self.observe(self.start + dt), [])

    def test_people_still_present_after_the_deadline_are_stranded(self):
        deadline = self.start + 180
        self.assertEqual(self.observe(deadline + 1), [], "confirmation period")
        found = self.observe(deadline + 1 + E.STRANDED_CONFIRM_SECONDS)
        self.assertEqual(len(found), 1)
        meta = found[0].metadata()
        self.assertEqual(meta["persons_remaining"], 2)
        self.assertEqual(meta["session_uid"], "EVAC-1")
        self.assertEqual(found[0].bbox, [10, 10, 260, 170], "evidence = union of the people left")

    def test_repeats_while_people_remain(self):
        deadline = self.start + 180
        self.observe(deadline + 1)
        self.assertEqual(len(self.observe(deadline + 10)), 1)
        self.assertEqual(self.observe(deadline + 30), [])
        self.assertEqual(len(self.observe(deadline + 10 + E.STRANDED_REPEAT_SECONDS)), 1)

    def test_empty_room_after_the_deadline_raises_nothing(self):
        for dt in range(181, 400, 5):
            self.assertEqual(self.observe(self.start + dt, people=[]), [])

    def test_each_camera_reports_what_it_sees_and_disarmed_cameras_are_silent(self):
        deadline = self.start + 180
        for cam in ("camera_01", "camera_02", "camera_03"):
            self.observe(deadline + 1, cam=cam)
        self.assertEqual(len(self.observe(deadline + 10, cam="camera_02")), 1)
        self.assertEqual(self.observe(deadline + 10, cam="camera_03"), [])

    def test_ended_session_stops_monitoring(self):
        self.m.set_sessions({"sessions": []}, now=self.start + 200)
        self.assertEqual(self.observe(self.start + 300), [])

    def test_stale_sessions_are_dropped_after_a_long_outage(self):
        later = self.start + E.EVAC_STALE_SECONDS + 1
        self.assertEqual(self.m.active_sessions(now=later), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
