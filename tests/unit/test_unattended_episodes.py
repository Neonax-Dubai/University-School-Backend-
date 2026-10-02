"""
Unattended objects are PHYSICAL EPISODES, not tracker ids: one object left in one place raises one
alert however many track ids it passes through, and reports its end once.

Deterministic: the frame clock is passed in, synthetic tracks, no model, no camera. The frame size and
boxes are C101's (2592x1944: the person-near radius is 12 % of the diagonal, ~389 px).
"""
import unittest
from unittest import mock

import helpers  # noqa: F401

import abandoned as A
import events

W, H = 2592, 1944
CAM, CAM2 = "camera_01", "camera_02"
RW, AR = A.RECOVERY_WINDOW_SECONDS, A.ATTENDED_RESOLVE_SECONDS   # 120 s, 300 s by default
BAG = [1200, 900, 1330, 1060]                 # a backpack on the floor, ~130x160 px
FAR_BAG = [300, 1500, 430, 1660]              # another resting place, far from BAG
NEAR_PERSON = [1250, 500, 1450, 1100]         # standing beside the bag
FAR_PERSON = [2200, 200, 2400, 800]           # elsewhere in the room
OCCLUDER = [1150, 700, 1380, 1300]            # standing IN FRONT of the bag (covers it)


class Obj:
    def __init__(self, track_id, bbox=BAG, class_id=24, class_name="backpack", confidence=0.5, camera_id=CAM):
        self.track_id, self.bbox, self.class_id, self.class_name = track_id, list(bbox), class_id, class_name
        self.confidence, self.camera_id, self.group = confidence, camera_id, "object"


def shifted(box, dx=0, dy=0):
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


class Scene:
    """Drives one detector at 10 fps on an explicit clock and records what it emits."""

    def __init__(self, cameras=(CAM,)):
        self.det = A.AbandonedObjectDetector({c: {"abandoned_object": True} for c in cameras}, dwell_seconds=30.0)
        self.t = 1000.0
        self.alerts, self.resolutions = [], []

    def run(self, seconds, objects=lambda t: [], persons=lambda t: [], camera=CAM, fps=10):
        start = self.t
        for _ in range(int(round(seconds * fps))):
            self.t += 1.0 / fps
            e = self.t - start
            for obj, meta in self.det.update(camera, objects(e), persons(e), W, H, now=self.t):
                self.alerts.append((self.t, camera, obj.track_id, meta))
            self.resolutions += [(self.t, camera, o, m) for o, m in self.det.drain_resolutions(camera)]
        return self

    def episode_ids(self):
        return [m["episode_id"] for *_, m in self.alerts]

    def state(self, episode_id):
        return next(e for e in self.det.episodes() if e["episode_id"] == episode_id)


class EpisodeLifecycleTests(unittest.TestCase):
    # 1
    def test_a_normal_unattended_object_alerts_once_after_the_dwell(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        self.assertEqual(len(s.alerts), 1)
        t, _, track, meta = s.alerts[0]
        self.assertAlmostEqual(t - 1000.0, 32.0, delta=0.3, msg="2 s owner grace + 30 s dwell")
        self.assertEqual((track, meta["episode_state"], meta["track_ids"], meta["track_changes"]),
                         ("O-1", "alerted", ["O-1"], 0))
        self.assertTrue(meta["episode_id"].startswith("UA-"))
        self.assertEqual(s.state(meta["episode_id"])["state"], A.ALERTED)

    # 2, 3, 17
    def test_an_object_left_for_half_an_hour_raises_one_alert(self):
        s = Scene().run(1800, lambda e: [Obj("O-1")], lambda e: [FAR_PERSON])
        self.assertEqual(len(s.alerts), 1, "same track, still unattended: no repeat")
        self.assertEqual(s.resolutions, [])
        self.assertEqual(s.state(s.episode_ids()[0])["state"], A.ALERTED)

    # 4 - the production O-101W-0940 -> 0956 -> 0975 -> 0994 -> 1011 pattern
    def test_track_ids_changing_under_occlusion_stay_one_episode(self):
        ids = ["O-0940", "O-0956", "O-0975", "O-0994", "O-1011"]

        def objects(e):                          # 50 s per id, then 6 s behind a person
            k, phase = int(e // 56), e % 56
            return [] if phase >= 50 else [Obj(ids[min(k, 4)])]

        def persons(e):
            return [OCCLUDER] if e % 56 >= 50 else [FAR_PERSON]
        s = Scene().run(280, objects, persons)
        self.assertEqual(len(s.alerts), 1)
        episode = s.state(s.episode_ids()[0])
        self.assertEqual(episode["track_ids"], ids)
        self.assertEqual(episode["state"], A.ALERTED)

    # 5
    def test_a_new_track_near_the_old_object_continues_its_episode(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        first = s.episode_ids()[0]
        s.run(10)                                # missed for 10 s, view clear
        s.run(60, lambda e: [Obj("O-2", shifted(BAG, 20, 15), class_id=26, class_name="handbag")])
        self.assertEqual(len(s.alerts), 1, "the same bag under a new id (and the other bag class)")
        self.assertEqual(s.state(first)["track_ids"], ["O-1", "O-2"])

    # 6, 14
    def test_a_new_track_far_away_is_a_separate_episode(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(40, lambda e: [Obj("O-1"), Obj("O-2", FAR_BAG)])
        self.assertEqual(len(s.alerts), 2)
        self.assertNotEqual(*s.episode_ids())
        self.assertEqual([m["track_ids"] for *_, m in s.alerts], [["O-1"], ["O-2"]])

    # 7
    def test_an_object_missed_for_a_while_and_back_continues_without_a_new_alert(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(RW - 15)                           # unseen with the spot in view, inside the window
        s.run(60, lambda e: [Obj("O-7")])
        self.assertEqual((len(s.alerts), s.resolutions), (1, []))

    # 8 - removal needs the whole window, never one frame
    def test_a_removed_object_resolves_after_the_recovery_window(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(0.1)
        self.assertEqual(s.resolutions, [], "one missing frame is not a removal")
        s.run(RW - 0.5)
        self.assertEqual(s.resolutions, [], "just inside the window")
        s.run(1.0)
        self.assertEqual(len(s.resolutions), 1)
        _, _, obj, meta = s.resolutions[0]
        self.assertEqual((meta["resolution"], meta["recovered"], meta["episode_id"], obj.track_id),
                         ("removed", True, s.episode_ids()[0], "O-1"))
        self.assertEqual(meta["video"], {"required": False})

    # 9
    def test_an_object_placed_again_later_is_a_new_episode_with_a_new_alert(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(RW + 10)                           # removed
        s.run(40, lambda e: [Obj("O-9")])        # placed again at the same place
        self.assertEqual(len(s.alerts), 2)
        self.assertNotEqual(*s.episode_ids())
        self.assertEqual(s.state(s.episode_ids()[0])["resolution"], "removed")

    # 10
    def test_a_person_standing_in_front_of_an_alerted_object_changes_nothing(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(20, persons=lambda e: [OCCLUDER])  # bag hidden for 20 s behind someone
        s.run(60, lambda e: [Obj("O-2")])        # reappears under a new id
        self.assertEqual((len(s.alerts), s.resolutions), (1, []))

    def test_time_behind_a_person_does_not_count_toward_removal(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        # unseen for 1.5 windows: people in front of it for 20 s at a time, the spot in view 3 s in between
        s.run(1.5 * RW, persons=lambda e: [OCCLUDER] if e % 23 < 20 else [])
        self.assertEqual(s.resolutions, [], "only ~13 % of it had the spot in view")

    # 11 (after the alert: hysteresis)
    def test_a_person_briefly_approaching_an_alerted_object_neither_closes_nor_re_raises_it(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        for _ in range(5):                       # five 8 s visits, a minute apart
            s.run(8, lambda e: [Obj("O-1")], lambda e: [NEAR_PERSON])
            s.run(60, lambda e: [Obj("O-1")])
        self.assertEqual((len(s.alerts), s.resolutions), (1, []))

    # 11 (before the alert: the original rule holds - a person near the object holds its clock)
    def test_before_the_alert_a_person_near_the_object_restarts_its_clock(self):
        s = Scene().run(25, lambda e: [Obj("O-1")])
        s.run(3, lambda e: [Obj("O-1")], lambda e: [NEAR_PERSON])
        s.run(30, lambda e: [Obj("O-1")])
        self.assertEqual(s.alerts, [], "28 s unattended in total, but not 30 s in a row")
        s.run(5, lambda e: [Obj("O-1")])
        self.assertEqual(len(s.alerts), 1)

    # 12
    def test_a_person_staying_beside_an_alerted_object_resolves_it_as_attended(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(AR - 1, lambda e: [Obj("O-1")], lambda e: [NEAR_PERSON])
        self.assertEqual(s.resolutions, [])
        s.run(2, lambda e: [Obj("O-1")], lambda e: [NEAR_PERSON])
        self.assertEqual([m["resolution"] for *_, m in s.resolutions], ["attended"])
        s.run(60, lambda e: [Obj("O-1")], lambda e: [NEAR_PERSON])
        self.assertEqual(len(s.alerts), 1, "attended: no alert while they stay")
        s.run(40, lambda e: [Obj("O-1")])        # they leave it behind again
        self.assertEqual(len(s.alerts), 2, "a new unattended episode, after the last one resolved")
        self.assertNotEqual(*s.episode_ids())

    def test_a_carried_bag_never_alerts(self):
        s = Scene().run(120, lambda e: [Obj("O-1", shifted(BAG, int(e * 15)))], lambda e: [shifted(NEAR_PERSON, int(e * 15))])
        self.assertEqual(s.alerts, [])

    # 13
    def test_several_objects_in_one_camera_are_separate_episodes(self):
        third = [1900, 1200, 2030, 1360]
        s = Scene().run(40, lambda e: [Obj("O-1"), Obj("O-2", FAR_BAG, 28, "suitcase"), Obj("O-3", third, 26, "handbag")])
        self.assertEqual(len(s.alerts), 3)
        self.assertEqual(len(set(s.episode_ids())), 3)
        self.assertEqual(sorted(m["object_type"] for *_, m in s.alerts), ["backpack", "handbag", "suitcase"])

    # 15
    def test_cameras_are_independent(self):
        s = Scene(cameras=(CAM, CAM2))
        for _ in range(400):                     # the same pixels in two cameras, 40 s
            s.t += 0.1
            for camera, track in ((CAM, "O-A"), (CAM2, "O-B")):
                for obj, meta in s.det.update(camera, [Obj(track, camera_id=camera)], [], W, H, now=s.t):
                    s.alerts.append((s.t, camera, obj.track_id, meta))
        self.assertEqual(sorted((c, m["track_ids"][0]) for _, c, _, m in s.alerts), [(CAM, "O-A"), (CAM2, "O-B")])
        self.assertNotEqual(*s.episode_ids())
        s.run(RW + 10, camera=CAM2)              # camera_02's bag removed
        self.assertEqual([(c, m["resolution"]) for _, c, _, m in s.resolutions], [(CAM2, "removed")])
        cam1 = [e for e in s.det.episodes(CAM) if e["state"] != A.RESOLVED]
        self.assertEqual([e["state"] for e in cam1], [A.ALERTED], "camera_01's episode is untouched")

    def test_a_track_from_another_camera_never_continues_an_episode(self):
        s = Scene(cameras=(CAM, CAM2)).run(40, lambda e: [Obj("O-1")])
        s.run(5, lambda e: [Obj("O-1", camera_id=CAM2)], camera=CAM2)
        self.assertEqual(len(s.det.episodes(CAM2)), 1)
        self.assertEqual(s.det.episodes(CAM2)[0]["track_ids"], ["O-1"])
        self.assertNotEqual(s.det.episodes(CAM2)[0]["episode_id"], s.episode_ids()[0])

    # 16
    def test_the_resolution_is_reported_exactly_once_and_names_the_alerts_episode(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(RW + 5)
        self.assertEqual(len(s.resolutions), 1)
        self.assertEqual(s.det.drain_resolutions(CAM), [])
        _, _, obj, meta = s.resolutions[0]
        alert = s.alerts[0][3]
        self.assertEqual((meta["episode_id"], meta["track_ids"], meta["episode_state"]),
                         (alert["episode_id"], ["O-1"], "resolved"))
        self.assertEqual(obj.bbox, BAG)
        s.run(300)
        self.assertEqual(len(s.resolutions), 1)


class MovementAndGeometryTests(unittest.TestCase):
    def test_an_alerted_object_picked_up_and_carried_away_resolves_as_moved(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(5, lambda e: [Obj("O-1", shifted(BAG, 300 + int(e * 60)))], lambda e: [NEAR_PERSON])
        self.assertEqual([m["resolution"] for *_, m in s.resolutions], ["moved"])
        self.assertEqual(len(s.alerts), 1)

    def test_an_object_moving_off_its_spot_as_the_dwell_ends_does_not_alert(self):
        s = Scene().run(31.5, lambda e: [Obj("O-1")])
        s.run(1.5, lambda e: [Obj("O-1", shifted(BAG, 400 + int(e * 40)))])
        self.assertEqual(s.alerts, [], "it was being moved when its 32 s came up")

    def test_a_box_that_wanders_with_nobody_near_is_not_a_move(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(10, lambda e: [Obj("O-1", shifted(BAG, 400))])              # 10 s off its spot, nobody there
        s.run(60, lambda e: [Obj("O-1")])
        self.assertEqual((len(s.alerts), s.resolutions), (1, []), "objects at rest do not move by themselves")

    def test_a_bag_relocated_by_a_visitor_ends_its_episode_at_once(self):
        # production 14:51 (camera_01): someone handles an ALERTED bag and leaves it ~0.6 box diagonals away
        relocated = shifted(BAG, 120, 80)
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(5, persons=lambda e: [NEAR_PERSON])                          # visitor in front of it
        s.run(3, lambda e: [Obj("O-2", relocated)], lambda e: [NEAR_PERSON])
        self.assertEqual([m["resolution"] for *_, m in s.resolutions], ["moved"], "closed now, not 120 s later")
        s.run(40, lambda e: [Obj("O-2", relocated)])                       # left there, unattended
        self.assertEqual(len(s.alerts), 2, "a new resting spot after a person moved it")
        self.assertEqual(len([e for e in s.det.episodes() if e["state"] == A.ALERTED]), 1, "never two open at once")

    def test_a_nudge_within_the_resting_tolerance_is_the_same_episode(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(20, persons=lambda e: [NEAR_PERSON])                         # a 20 s visit, bag hidden
        s.run(60, lambda e: [Obj("O-3", shifted(BAG, 40, 30))])            # back, a little lower
        self.assertEqual((len(s.alerts), s.resolutions), (1, []))

    def test_one_bad_box_is_not_a_move(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(0.5, lambda e: [Obj("O-1", shifted(BAG, 400))], lambda e: [NEAR_PERSON])   # even with someone there
        s.run(60, lambda e: [Obj("O-1")])
        self.assertEqual((len(s.alerts), s.resolutions), (1, []))

    def test_a_second_box_on_the_same_bag_is_a_fragment_not_an_episode(self):
        partial = [1200, 980, 1290, 1060]        # a lower part of the bag, as another track
        s = Scene().run(60, lambda e: [Obj("O-1"), Obj("O-2", partial, 26, "handbag", 0.3)])
        self.assertEqual(len(s.alerts), 1)
        self.assertEqual(len(s.det.episodes(CAM)), 1)

    def test_a_much_smaller_box_nearby_is_not_the_same_object(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(5)
        s.run(40, lambda e: [Obj("O-2", [1240, 950, 1270, 980])])   # 30x30: a phone-sized thing
        self.assertEqual(len(s.alerts), 2)

    def test_strict_class_mode_keeps_bag_classes_apart(self):
        with mock.patch.object(A, "ASSOCIATION_STRICT_CLASS", True):
            s = Scene().run(40, lambda e: [Obj("O-1")])
            s.run(5)
            s.run(40, lambda e: [Obj("O-2", class_id=26, class_name="handbag")])
        self.assertEqual(len(s.alerts), 2)

    def test_a_camera_stall_is_not_observed_absence(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.t += 300.0                             # no frames at all for five minutes
        s.run(3)                                 # then the first frames miss the bag
        s.run(5, lambda e: [Obj("O-1")])
        self.assertEqual((len(s.alerts), s.resolutions), (1, []))

    def test_a_disabled_camera_does_nothing(self):
        det = A.AbandonedObjectDetector({CAM: {"abandoned_object": False}})
        self.assertEqual(det.update(CAM, [Obj("O-1")], [], W, H, now=1.0), [])

    def test_resolved_episodes_are_forgotten_after_retention(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(RW + 10)
        self.assertEqual(len(s.det.episodes()), 1)
        s.run(A.RESOLVED_RETENTION_SECONDS + 40)
        self.assertEqual(s.det.episodes(), [])


class ClassConfigTests(unittest.TestCase):
    def test_names_ids_and_luggage_are_accepted(self):
        self.assertEqual(A.parse_classes("backpack, 26 ,luggage", log=lambda m: None), frozenset({24, 26, 28}))

    def test_unsupported_classes_are_refused_never_enabled(self):
        logged = []
        self.assertEqual(A.parse_classes("backpack,box,knife,43,67", log=logged.append), frozenset({24}))
        self.assertIn("box", logged[0])

    def test_nothing_valid_falls_back_to_the_validated_default(self):
        logged = []
        self.assertEqual(A.parse_classes("package", log=logged.append), frozenset({24, 26, 28}))
        self.assertEqual(len(logged), 2)

    def test_the_default_is_the_three_bag_classes(self):
        self.assertEqual(A.CANDIDATE_CLASS_IDS, frozenset({24, 26, 28}))

    def test_a_disabled_class_is_ignored(self):
        with mock.patch.object(A, "CANDIDATE_CLASS_IDS", frozenset({28})):
            s = Scene().run(40, lambda e: [Obj("O-1")])
        self.assertEqual(s.alerts, [])


class Sender:
    def __init__(self):
        self.sent = []

    def submit(self, event, notable=False):
        self.sent.append(event)


class EventLayerTests(unittest.TestCase):
    def pipeline(self):
        with mock.patch.object(events, "EVENTS_ENABLED", False):
            p = events.EventPipeline("http://dashboard.invalid", "x")
        p.sender = Sender()
        return p

    def test_the_alert_and_its_resolution_travel_as_one_episode(self):
        s = Scene().run(40, lambda e: [Obj("O-1")])
        s.run(5, lambda e: [Obj("O-2")])
        s.run(RW + 10)
        p = self.pipeline()
        _, _, track, meta = s.alerts[0]
        self.assertEqual(p.handle_abandoned_event(Obj(track), meta, W, H, 1.0), "sent")
        _, _, obj, rmeta = s.resolutions[0]
        self.assertEqual(p.handle_abandoned_resolution(obj, rmeta, W, H, 2.0), "sent")
        alert, resolution = (events.zayed_enrich(e) for e in p.sender.sent)
        self.assertEqual({alert["event_type"], resolution["event_type"]}, {"UNATTENDED_OBJECT_DETECTED"})
        self.assertEqual(alert["metadata"]["episode_id"], resolution["metadata"]["episode_id"])
        self.assertNotIn("recovered", alert["metadata"])
        self.assertIs(resolution["metadata"]["recovered"], True)
        self.assertEqual(resolution["metadata"]["track_ids"], ["O-1", "O-2"])

    def test_the_debounce_is_per_episode_not_per_track(self):
        p = self.pipeline()
        meta = {"episode_id": "UA-TEST-0001", "dwell_seconds": 30.0}
        self.assertEqual(p.handle_abandoned_event(Obj("O-1"), dict(meta), W, H, 1.0), "sent")
        self.assertEqual(p.handle_abandoned_event(Obj("O-2"), dict(meta), W, H, 2.0), "suppressed",
                         "the same episode under another track id is the same alert")
        other = {"episode_id": "UA-TEST-0002", "dwell_seconds": 30.0}
        self.assertEqual(p.handle_abandoned_event(Obj("O-1"), other, W, H, 3.0), "sent",
                         "another episode - even on the same track - is never suppressed by this one")
        self.assertEqual(len(p.sender.sent), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
