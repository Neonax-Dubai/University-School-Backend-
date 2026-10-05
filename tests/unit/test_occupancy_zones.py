"""
Classroom occupancy counted in OCCUPANCY ownership zones (the dashboard's Zone, rules["occupancy"]).

C101: camera_01 and camera_02 each own one non-overlapping part of the room; camera_03 (the
faculty view) owns none and is never counted. The overcrowding cases at 40/41/42 are LOGIC tests on
synthetic tracks - no camera ever saw 42 people.
"""
import unittest

import helpers  # noqa: F401
from helpers import Config, Track

import events
import occupancy as O

W, H = 1000, 800                                      # frame size for the synthetic boxes
LEFT = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]      # left half of the frame
RIGHT = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]


def zone(polygon, occupancy=True, overcrowding=True, enabled=True, **rules):
    return {"id": 1, "name": "z", "type": "custom", "enabled": enabled, "coordinates": polygon,
            "rules": dict(rules, occupancy=occupancy, overcrowding=overcrowding)}


def config(camera_id, zones=(), capacity=41, **kwargs):
    c = Config(camera_id, capacity=capacity, **kwargs)
    c.zones = list(zones)
    return c


def person(i, cx, cy, size=40):
    """A person box centred on (cx, cy), in pixels."""
    return Track(f"P-TEST-{i:04d}", [cx - size / 2, cy - size, cx + size / 2, cy + size])


def people(n, cx=250, cy=400):
    return [person(i, cx + (i % 10) * 10, cy + (i // 10) * 10) for i in range(n)]


def c101(overcrowding=True):
    occ = O.ClassroomOccupancy()
    occ.configure([config("camera_01", [zone(LEFT, overcrowding=overcrowding)]),
                   config("camera_02", [zone(LEFT, overcrowding=overcrowding)]),
                   config("camera_03")])                 # faculty view: no occupancy zone
    return occ


def run(occ, frames, start=0.0, seconds=12, step=0.5):
    """Feed the same per-camera tracks for `seconds`; return (last snapshot row, event types)."""
    events_seen, t = [], start
    while t < start + seconds:
        for camera_id, tracks in frames.items():
            occ.observe(camera_id, tracks, t, W, H)
        events_seen += [e[0] for e in occ.decisions(t)]
        t += step
    return occ.snapshot(t)[0], events_seen


class ZoneOwnershipCountTests(unittest.TestCase):
    def test_the_classroom_is_the_sum_of_the_owned_zones_and_camera_03_is_excluded(self):
        row, _ = run(c101(), {"camera_01": people(2), "camera_02": people(1), "camera_03": people(5)})
        self.assertEqual((row["occupancy"], row["method"]), (3, "zone_ownership"), "2 + 1, not 2 + 1 + 5")
        self.assertEqual(row["per_camera"], {"camera_01": 2, "camera_02": 1}, "camera_03 is not a contributor")

    def test_people_outside_the_ownership_zone_are_not_counted(self):
        outside = [person(9, 800, 400)]                  # centre in the right half: not camera_01's zone
        row, _ = run(c101(), {"camera_01": people(2) + outside, "camera_02": [], "camera_03": []})
        self.assertEqual(row["occupancy"], 2)

    def test_a_track_counts_once_even_inside_two_zones_of_its_camera(self):
        occ = O.ClassroomOccupancy()
        occ.configure([config("camera_01", [zone(LEFT), zone([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])])])
        row, _ = run(occ, {"camera_01": people(1)})
        self.assertEqual(row["occupancy"], 1)

    def test_one_person_stays_one_count_across_frames(self):
        row, _ = run(c101(), {"camera_01": people(1), "camera_02": [], "camera_03": []}, seconds=60)
        self.assertEqual((row["occupancy"], row["occupancy_max"]), (1, 1), "tracks, not detections")

    def test_an_empty_room_is_zero_and_the_count_falls_when_people_leave(self):
        occ = c101()
        row, _ = run(occ, {"camera_01": people(2), "camera_02": people(1), "camera_03": []})
        self.assertEqual(row["occupancy"], 3)
        row, _ = run(occ, {"camera_01": [], "camera_02": [], "camera_03": []}, start=12, seconds=11)
        self.assertEqual((row["occupancy"], row["per_camera"]), (0, {"camera_01": 0, "camera_02": 0}))

    def test_the_zone_convention_is_the_track_anchor_the_box_centre(self):
        # A box straddling the zone edge counts by its centre, as every other zone rule does.
        straddling_in = Track("P-EDGE-0001", [440, 300, 540, 500])     # centre x 490 < 500
        straddling_out = Track("P-EDGE-0002", [460, 300, 560, 500])    # centre x 510 > 500
        row, _ = run(c101(), {"camera_01": [straddling_in, straddling_out], "camera_02": [], "camera_03": []})
        self.assertEqual(row["occupancy"], 1)

    def test_a_person_moving_between_ownership_zones_is_never_counted_twice(self):
        occ = c101()
        totals, t = [], 0.0
        for step in range(60):                          # 30 s: camera_01's zone, then camera_02's
            in_first = step < 30
            occ.observe("camera_01", people(1) if in_first else [], t, W, H)
            occ.observe("camera_02", [] if in_first else people(1), t, W, H)
            occ.observe("camera_03", [], t, W, H)
            totals.append(occ.snapshot(t)[0]["occupancy"])
            t += 0.5
        self.assertLessEqual(max(totals), 1, totals)
        self.assertEqual(totals[-1], 1)

    def test_a_missing_owner_camera_is_reported_missing_and_camera_03_never_is(self):
        occ = c101()
        for t in range(12):
            occ.observe("camera_01", people(2), float(t), W, H)
            occ.observe("camera_03", people(4), float(t), W, H)
        row = occ.snapshot(12.0)[0]
        self.assertEqual((row["occupancy"], row["per_camera"]), (2, {"camera_01": 2, "camera_02": None}),
                         "camera_02 sent no frames: a partial count, flagged by its null")

    def test_a_disabled_zone_or_one_without_the_occupancy_rule_owns_nothing(self):
        occ = O.ClassroomOccupancy()
        occ.configure([config("camera_01", [zone(LEFT, enabled=False)]),
                       config("camera_02", [zone(LEFT, occupancy=False, crowd_detection=True)])])
        row, _ = run(occ, {"camera_01": people(2), "camera_02": people(1)})
        self.assertEqual((row["occupancy"], row["method"]), (2, "max_camera"), "no ownership zone: unchanged")

    def test_without_occupancy_zones_the_classroom_keeps_max_camera(self):
        occ = O.ClassroomOccupancy()
        occ.configure([config(c) for c in ("camera_01", "camera_02", "camera_03")])
        row, _ = run(occ, {"camera_01": people(2), "camera_02": people(1), "camera_03": people(3)})
        self.assertEqual((row["occupancy"], row["method"]), (3, "max_camera"))

    def test_the_percentage_uses_the_capacity(self):
        occ = c101()
        for camera_id, tracks in {"camera_01": people(2), "camera_02": people(1), "camera_03": people(5)}.items():
            occ.observe(camera_id, tracks, 0.0, W, H)
        [update] = [meta for kind, _, meta in occ.decisions(0.0) if kind == "OCCUPANCY_UPDATED"]
        self.assertEqual((update["occupancy"], update["capacity"], update["occupancy_pct"]), (3, 41, 7.3))


class OvercrowdingLogicTests(unittest.TestCase):
    """LOGIC tests at capacity 41, threshold 100 % (occupancy > capacity). Synthetic tracks."""

    def split(self, n):
        return {"camera_01": people(n - n // 2), "camera_02": people(n // 2), "camera_03": people(7)}

    def test_40_and_41_do_not_overcrowd_42_does_after_the_hold(self):
        for n, expected in ((40, 0), (41, 0), (42, 1)):
            row, seen = run(c101(), self.split(n), seconds=45)
            self.assertEqual(row["occupancy"], n)
            self.assertEqual(seen.count("OVERCROWDING_DETECTED"), expected, n)

    def test_42_must_hold_for_30_seconds(self):
        _, seen = run(c101(), self.split(42), seconds=30)
        self.assertEqual(seen.count("OVERCROWDING_DETECTED"), 0, "not before the 30 s hold (+ window)")

    def test_one_event_per_episode_then_re_armed_after_60_seconds_below(self):
        occ = c101()
        _, first = run(occ, self.split(42), seconds=120)
        self.assertEqual(first.count("OVERCROWDING_DETECTED"), 1, "no flood while it persists")
        _, short_dip = run(occ, self.split(30), start=120, seconds=40)
        _, again = run(occ, self.split(42), start=160, seconds=60)
        self.assertEqual((short_dip + again).count("OVERCROWDING_DETECTED"), 0, "a 40 s dip does not re-arm")
        _, long_dip = run(occ, self.split(30), start=220, seconds=80)
        _, next_episode = run(occ, self.split(42), start=300, seconds=60)
        self.assertEqual(next_episode.count("OVERCROWDING_DETECTED"), 1, "re-armed after 60 s below")

    def test_occupancy_zones_without_the_overcrowding_use_case_never_alarm(self):
        _, seen = run(c101(overcrowding=False), self.split(45), seconds=60)
        self.assertEqual(seen.count("OVERCROWDING_DETECTED"), 0)

    def test_camera_03_cannot_push_the_classroom_over(self):
        _, seen = run(c101(), {"camera_01": people(20), "camera_02": people(20), "camera_03": people(50)},
                      seconds=60)
        self.assertEqual(seen.count("OVERCROWDING_DETECTED"), 0, "40 counted; the faculty view is not")

    def test_an_outage_restarts_the_hold(self):
        occ = c101()
        _, before = run(occ, self.split(42), seconds=15)          # over for 15 s
        during = []
        for t in range(15, 55):                          # every camera silent for 40 s: unknown
            during += [e[0] for e in occ.decisions(float(t))]
        _, after = run(occ, self.split(42), start=55, seconds=25)
        self.assertEqual((before + during + after).count("OVERCROWDING_DETECTED"), 0,
                         "15 s before + 25 s after an outage is not 30 s: the hold does not span it")
        _, later = run(occ, self.split(42), start=80, seconds=10)
        self.assertEqual(later.count("OVERCROWDING_DETECTED"), 1, "30 s held after the outage")


class CrowdZoneTests(unittest.TestCase):
    def test_a_zone_crowd_is_no_longer_reported_as_classroom_overcrowding(self):
        event = events.zayed_enrich({"event_type": "crowd_detected", "camera_id": "camera_01", "metadata": {}})
        self.assertEqual(event["event_type"], "crowd_detected")
        self.assertEqual(events.zayed_enrich({"event_type": "fall_detected", "metadata": {}})["event_type"],
                         "FALL_DETECTED", "the other mappings are unchanged")


if __name__ == "__main__":
    unittest.main(verbosity=2)
