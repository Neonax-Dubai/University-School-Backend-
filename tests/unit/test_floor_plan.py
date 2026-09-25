import unittest

import helpers  # noqa: F401
from helpers import Config, Track

import floor_plan as F
import occupancy as O

# 1 px = 1 cm: image (0..800, 0..600) -> floor (0..8 m, 0..6 m)
SCALE = {"image_points": [[0, 0], [800, 0], [800, 600], [0, 600]],
         "floor_points_m": [[0, 0], [8, 0], [8, 6], [0, 6]]}
PLAN = {"width_m": 8, "height_m": 6, "grid_m": 0.5}
#: The heatmap switch, as the backend sends it in the camera payload. These tests predate the
#: switch being enforced, so they armed nothing; arming it here is what keeps them a statement
#: about the heatmap's behaviour rather than about the switch.
ON = {F.HEATMAP_FEATURE: True}


class HomographyTests(unittest.TestCase):
    def test_valid_calibration_projects_foot_points(self):
        m = F.build_homography(SCALE)
        x, y = F.project(m, [(400, 300)])[0]
        self.assertAlmostEqual(x, 4.0, places=3)
        self.assertAlmostEqual(y, 3.0, places=3)

    def test_missing_or_degenerate_calibration_is_unmapped(self):
        self.assertIsNone(F.build_homography({}))
        self.assertIsNone(F.build_homography({"image_points": [[0, 0]] * 4, "floor_points_m": [[0, 0]] * 4}))
        self.assertIsNone(F.build_homography({"image_points": [[0, 0], [1, 0], [1, 1]],
                                              "floor_points_m": [[0, 0], [1, 0], [1, 1]]}))


class FusionTests(unittest.TestCase):
    def test_same_person_seen_by_two_cameras_is_one(self):
        people = F.fuse([("camera_01", 2.0, 2.0), ("camera_02", 2.3, 2.1), ("camera_03", 6.0, 1.0)])
        self.assertEqual(len(people), 2)

    def test_two_people_side_by_side_on_one_camera_stay_two(self):
        self.assertEqual(len(F.fuse([("camera_01", 2.0, 2.0), ("camera_01", 2.3, 2.0)])), 2)


class MapperTests(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self.m = F.FloorPlanMapper(log=self.logs.append)
        self.configs = [Config(c, homography=SCALE, floor_plan=PLAN, features=dict(ON))
                        for c in ("camera_01", "camera_02", "camera_03")]
        self.m.configure(self.configs, now=0.0)

    def step(self, t, per_camera):
        for cam, tracks in per_camera.items():
            self.m.observe(cam, tracks, t)
        self.m.tick(t)

    def test_fused_occupancy_and_heatmap(self):
        # Points kept off the 0.5 m cell boundaries (a boundary point may land either side).
        a1 = Track("P-1", [190, 110, 230, 210])     # foot (210, 210) -> (2.1, 2.1) m
        a2 = Track("P-7", [195, 115, 235, 215])     # the same student from camera_02 -> (2.15, 2.15)
        b = Track("P-2", [590, 25, 630, 125])       # another student -> (6.1, 1.25)
        for t in range(1, 11):
            self.step(float(t), {"camera_01": [a1, b], "camera_02": [a2], "camera_03": []})
        self.assertEqual(self.m.fused_counts("C101", 10.0, 10.0)[-1], 2)
        rows = self.m.take_heatmaps(now=10.0)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["cols"], row["rows"]), (16, 12))
        cells = {(c, r): s for c, r, s in row["cells"]}
        self.assertIn((4, 4), cells)              # (2.1, 2.1) m in 0.5 m cells
        self.assertIn((12, 2), cells)             # (6.1, 1.25) m
        self.assertAlmostEqual(row["person_seconds"], 20.0, delta=2.1)
        self.assertEqual(self.m.take_heatmaps(now=11.0), [], "a new window starts empty")

    def test_location_for_an_event(self):
        loc = self.m.location_for("camera_01", [180, 100, 220, 200])
        self.assertEqual((loc["floor_x_m"], loc["floor_y_m"]), (2.0, 2.0))

    def test_partially_calibrated_classroom_keeps_max_camera(self):
        configs = [Config("camera_01", homography=SCALE, floor_plan=PLAN, features=dict(ON)),
                   Config("camera_02", features=dict(ON)), Config("camera_03", features=dict(ON))]
        self.m.configure(configs, now=0.0)
        self.step(1.0, {"camera_01": [Track("P-1", [180, 100, 220, 200])]})
        self.assertIsNone(self.m.fused_counts("C101", 1.0, 10.0))

    def test_occupancy_uses_floor_fusion_when_fully_mapped(self):
        occ = O.ClassroomOccupancy()
        occ.configure(self.configs)
        occ.set_fusion(self.m)
        a1, a2 = Track("P-1", [180, 100, 220, 200]), Track("P-7", [185, 105, 225, 205])
        for t in range(1, 6):
            occ.observe("camera_01", [a1], float(t))
            occ.observe("camera_02", [a2], float(t))
            occ.observe("camera_03", [], float(t))
            self.step(float(t), {"camera_01": [a1], "camera_02": [a2], "camera_03": []})
        row = occ.snapshot(now=5.0)[0]
        self.assertEqual(row["method"], "floor_fusion")
        self.assertEqual(row["occupancy"], 1)


class HeatmapFeatureSwitchTests(unittest.TestCase):
    """occupancy_heatmap must decide whether a heatmap exists - and decide nothing else.

    The switch was declared in floor_plan.py and never read, so a calibrated classroom
    accumulated and published person-seconds whether or not an operator had armed it.
    """

    def build(self, features):
        mapper = F.FloorPlanMapper(log=lambda *_: None)
        configs = [Config(c, homography=SCALE, floor_plan=PLAN, features=dict(features))
                   for c in ("camera_01", "camera_02", "camera_03")]
        mapper.configure(configs, now=0.0)
        return mapper, configs

    def run_people(self, mapper, seconds=10):
        """Two students, one of them seen by two cameras - the fixture the ON case uses."""
        a1 = Track("P-1", [190, 110, 230, 210])
        a2 = Track("P-7", [195, 115, 235, 215])
        b = Track("P-2", [590, 25, 630, 125])
        for t in range(1, seconds + 1):
            mapper.observe("camera_01", [a1, b], float(t))
            mapper.observe("camera_02", [a2], float(t))
            mapper.observe("camera_03", [], float(t))
            mapper.tick(float(t))
        return mapper

    # ------------------------------------------------------------------ OFF
    def test_off_generates_no_heatmap_at_all(self):
        mapper, _ = self.build({})
        self.assertEqual(mapper.stats()["heatmap_armed_classrooms"], [])
        self.assertEqual(mapper.stats()["heatmap_classrooms"], [],
                         "a grid was allocated for a classroom nobody armed")
        self.run_people(mapper)
        self.assertEqual(mapper.take_heatmaps(now=10.0), [],
                         "person-seconds were accumulated while the feature was off")

    def test_off_publishes_nothing(self):
        mapper, _ = self.build({})
        self.run_people(mapper)
        posted = []

        class _Session:
            def post(self, url, json=None, timeout=None):
                posted.append(json)
                raise AssertionError("the publisher posted with the feature off")

        publisher = F.HeatmapPublisher(mapper, "http://dashboard", "token", session=_Session(),
                                       log=lambda *_: None)
        self.assertFalse(publisher.publish_once(now=10.0), "publish_once claimed it sent something")
        self.assertEqual(posted, [])
        self.assertEqual(publisher.sent, 0)

    def test_one_camera_off_stops_that_classrooms_heatmap(self):
        # The heatmap is ONE fused grid per classroom; a single camera cannot be subtracted
        # from it without changing the fusion, so the room's heatmap stops.
        mapper = F.FloorPlanMapper(log=lambda *_: None)
        mapper.configure([Config("camera_01", homography=SCALE, floor_plan=PLAN, features=dict(ON)),
                          Config("camera_02", homography=SCALE, floor_plan=PLAN, features=dict(ON)),
                          Config("camera_03", homography=SCALE, floor_plan=PLAN, features={})], now=0.0)
        self.run_people(mapper)
        self.assertEqual(mapper.take_heatmaps(now=10.0), [])

    def test_turning_it_off_discards_the_open_window(self):
        mapper, _ = self.build(ON)
        self.run_people(mapper)                     # a part-window is now accumulating
        mapper.configure([Config(c, homography=SCALE, floor_plan=PLAN, features={})
                          for c in ("camera_01", "camera_02", "camera_03")], now=10.0)
        self.assertEqual(mapper.take_heatmaps(now=11.0), [],
                         "person-seconds gathered before the switch was turned off still escaped")

    # ------------------------------------------------------------------- ON
    def test_on_keeps_the_existing_behaviour(self):
        mapper, _ = self.build(ON)
        self.run_people(mapper)
        rows = mapper.take_heatmaps(now=10.0)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["cols"], row["rows"]), (16, 12))
        cells = {(c, r): s for c, r, s in row["cells"]}
        self.assertIn((4, 4), cells)
        self.assertIn((12, 2), cells)
        self.assertAlmostEqual(row["person_seconds"], 20.0, delta=2.1)
        self.assertEqual(row["method"], "floor_fusion")

    def test_turning_it_back_on_resumes(self):
        mapper, _ = self.build({})
        self.run_people(mapper)
        mapper.configure([Config(c, homography=SCALE, floor_plan=PLAN, features=dict(ON))
                          for c in ("camera_01", "camera_02", "camera_03")], now=10.0)
        for t in range(11, 21):
            mapper.observe("camera_01", [Track("P-1", [190, 110, 230, 210])], float(t))
            mapper.tick(float(t))
        rows = mapper.take_heatmaps(now=20.0)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["person_seconds"], 0)

    # --------------------------------------------------- nothing else moves
    def test_occupancy_and_fusion_are_untouched_by_the_switch(self):
        # The whole risk of this change: the switch must not reach the occupancy count, which
        # shares the projection and the floor plan with the heatmap.
        results = {}
        for label, features in (("on", ON), ("off", {})):
            mapper, configs = self.build(features)
            occ = O.ClassroomOccupancy()
            occ.configure(configs)
            occ.set_fusion(mapper)
            a1, a2 = Track("P-1", [180, 100, 220, 200]), Track("P-7", [185, 105, 225, 205])
            for t in range(1, 6):
                occ.observe("camera_01", [a1], float(t))
                occ.observe("camera_02", [a2], float(t))
                occ.observe("camera_03", [], float(t))
                mapper.observe("camera_01", [a1], float(t))
                mapper.observe("camera_02", [a2], float(t))
                mapper.observe("camera_03", [], float(t))
                mapper.tick(float(t))
            row = occ.snapshot(now=5.0)[0]
            results[label] = (row["method"], row["occupancy"],
                              mapper.fused_counts("C101", 5.0, 10.0),
                              mapper.mapped("camera_01"),
                              mapper.location_for("camera_01", [180, 100, 220, 200]))
        self.assertEqual(results["on"], results["off"],
                         "the heatmap switch changed occupancy, fusion or event location")
        self.assertEqual(results["off"][0], "floor_fusion")
        self.assertEqual(results["off"][1], 1)


class HeatmapSwitchBlastRadiusTests(unittest.TestCase):
    """What the switch is NOT allowed to reach.

    occupancy and location are covered by behaviour above. Camera health, person recognition
    and classroom presence cannot be covered that way from here - they are other processes -
    so they are covered structurally: floor_plan.py must not touch them at all. A future edit
    that wired the heatmap switch into any of them would have to import one of these first.
    """

    FORBIDDEN = ("camera_health", "face_id_manager", "known_person", "reid",
                 "presence", "events", "outbox", "evidence")

    def test_floor_plan_does_not_reach_into_any_other_subsystem(self):
        import inspect
        source = inspect.getsource(F)
        for name in self.FORBIDDEN:
            self.assertNotIn(f"import {name}", source,
                             f"floor_plan.py imported {name}: the heatmap switch can now "
                             f"affect it")

    def test_the_armed_set_is_the_only_new_state_the_switch_owns(self):
        mapper = F.FloorPlanMapper(log=lambda *_: None)
        configs = [Config(c, homography=SCALE, floor_plan=PLAN, features=dict(ON))
                   for c in ("camera_01", "camera_02", "camera_03")]
        mapper.configure(configs, now=0.0)
        armed = mapper.stats()["heatmap_armed_classrooms"]

        # Arming reports the room; it does not silently change what is mapped or planned.
        self.assertEqual(armed, ["C101"])
        self.assertEqual(mapper.stats()["mapped_cameras"], ["camera_01", "camera_02", "camera_03"])

        mapper.configure([Config(c, homography=SCALE, floor_plan=PLAN, features={})
                          for c in ("camera_01", "camera_02", "camera_03")], now=1.0)
        self.assertEqual(mapper.stats()["heatmap_armed_classrooms"], [])
        self.assertEqual(mapper.stats()["mapped_cameras"],
                         ["camera_01", "camera_02", "camera_03"],
                         "disarming the heatmap unmapped a camera - occupancy would change")

    def test_arming_alone_does_not_create_a_grid_without_a_floor_plan(self):
        # Both preconditions are independent: the switch says "allowed to", the floor plan
        # says "possible to". Neither implies the other.
        mapper = F.FloorPlanMapper(log=lambda *_: None)
        mapper.configure([Config(c, homography=SCALE, features=dict(ON))
                          for c in ("camera_01", "camera_02", "camera_03")], now=0.0)
        self.assertEqual(mapper.stats()["heatmap_armed_classrooms"], ["C101"])
        self.assertEqual(mapper.stats()["heatmap_classrooms"], [],
                         "a grid was allocated for a classroom with no floor plan")


class OccupancyTests(unittest.TestCase):
    def test_max_camera_never_double_counts_and_overcrowding_has_hysteresis(self):
        occ = O.ClassroomOccupancy()
        occ.configure([Config(c, capacity=2) for c in ("camera_01", "camera_02", "camera_03")])
        three = [Track(f"P-{i}", [0, 0, 1, 1]) for i in range(3)]
        events = []
        for t in range(0, 40):
            for cam in ("camera_01", "camera_02", "camera_03"):
                occ.observe(cam, three, float(t))
            events += [e[0] for e in occ.decisions(float(t))]
        self.assertEqual(occ.snapshot(40.0)[0]["occupancy"], 3, "3 people seen by 3 cameras = 3, not 9")
        self.assertEqual(events.count("OVERCROWDING_DETECTED"), 1)
        for t in range(40, 60):
            for cam in ("camera_01", "camera_02", "camera_03"):
                occ.observe(cam, three, float(t))
            events += [e[0] for e in occ.decisions(float(t))]
        self.assertEqual(events.count("OVERCROWDING_DETECTED"), 1, "no flapping while it persists")


if __name__ == "__main__":
    unittest.main(verbosity=2)
